// uring.h — a minimal io_uring wrapper on raw syscalls (no liburing: the serving image has the kernel headers
// but not the library). One ring per thread, no SQPOLL / IOPOLL (both need extra privileges or device poll
// queues); submissions are IORING_OP_READ with O_DIRECT fds into caller-aligned buffers, completions reaped by
// io_uring_enter(GETEVENTS). Used by the Engram v2 row fetcher (512-B / 1-KB random reads) and by the expert
// store's NVMe tail (4.7-MB record reads).
//
// Docker's default seccomp profile denies io_uring_setup / io_uring_enter / io_uring_register (EPERM): the
// launcher passes docker/sm120/seccomp-io_uring.json (the default profile + those three) when an NVMe mode is on.
#pragma once
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>

namespace uring {

static inline int sys_setup(unsigned entries, io_uring_params* p) { return (int)syscall(__NR_io_uring_setup, entries, p); }
static inline int sys_enter(int fd, unsigned to_submit, unsigned min_complete, unsigned flags) {
    return (int)syscall(__NR_io_uring_enter, fd, to_submit, min_complete, flags, nullptr, 0);
}

struct Ring {
    int fd = -1;
    unsigned entries = 0;
    // submission queue
    unsigned* sq_head = nullptr;
    unsigned* sq_tail = nullptr;
    unsigned* sq_mask = nullptr;
    unsigned* sq_array = nullptr;
    io_uring_sqe* sqes = nullptr;
    void* sq_ptr = nullptr;
    size_t sq_len = 0;
    size_t sqes_len = 0;
    // completion queue
    unsigned* cq_head = nullptr;
    unsigned* cq_tail = nullptr;
    unsigned* cq_mask = nullptr;
    io_uring_cqe* cqes = nullptr;
    void* cq_ptr = nullptr;
    size_t cq_len = 0;
    unsigned pending_sqes = 0;   // written but not yet submitted
    unsigned in_flight = 0;      // submitted, not yet reaped

    bool ok() const { return fd >= 0; }

    // returns 0, or -errno
    int init(unsigned n) {
        io_uring_params p;
        memset(&p, 0, sizeof(p));
        const int r = sys_setup(n, &p);
        if (r < 0) return -errno;
        fd = r;
        entries = p.sq_entries;
        sq_len = p.sq_off.array + p.sq_entries * sizeof(unsigned);
        cq_len = p.cq_off.cqes + p.cq_entries * sizeof(io_uring_cqe);
        if (p.features & IORING_FEAT_SINGLE_MMAP) {
            if (cq_len > sq_len) sq_len = cq_len;
            cq_len = sq_len;
        }
        sq_ptr = mmap(nullptr, sq_len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQ_RING);
        if (sq_ptr == MAP_FAILED) { const int e = errno; close(); return -e; }
        if (p.features & IORING_FEAT_SINGLE_MMAP) {
            cq_ptr = sq_ptr;
        } else {
            cq_ptr = mmap(nullptr, cq_len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_CQ_RING);
            if (cq_ptr == MAP_FAILED) { const int e = errno; close(); return -e; }
        }
        char* sq = (char*)sq_ptr;
        sq_head = (unsigned*)(sq + p.sq_off.head);
        sq_tail = (unsigned*)(sq + p.sq_off.tail);
        sq_mask = (unsigned*)(sq + p.sq_off.ring_mask);
        sq_array = (unsigned*)(sq + p.sq_off.array);
        sqes_len = p.sq_entries * sizeof(io_uring_sqe);
        sqes = (io_uring_sqe*)mmap(nullptr, sqes_len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQES);
        if (sqes == MAP_FAILED) { const int e = errno; sqes = nullptr; close(); return -e; }
        char* cq = (char*)cq_ptr;
        cq_head = (unsigned*)(cq + p.cq_off.head);
        cq_tail = (unsigned*)(cq + p.cq_off.tail);
        cq_mask = (unsigned*)(cq + p.cq_off.ring_mask);
        cqes = (io_uring_cqe*)(cq + p.cq_off.cqes);
        return 0;
    }

    void close() {
        if (sqes) munmap(sqes, sqes_len);
        if (cq_ptr && cq_ptr != sq_ptr) munmap(cq_ptr, cq_len);
        if (sq_ptr) munmap(sq_ptr, sq_len);
        if (fd >= 0) ::close(fd);
        sqes = nullptr; cq_ptr = sq_ptr = nullptr; fd = -1;
    }

    unsigned space() const { return entries - pending_sqes - in_flight; }

    // queue one read; the caller guarantees space() > 0. offset / len / buf aligned for O_DIRECT.
    void prep_read(int file_fd, void* buf, unsigned len, uint64_t offset, uint64_t user_data) {
        const unsigned tail = *sq_tail;   // single producer: a plain load of our own tail
        const unsigned idx = tail & *sq_mask;
        io_uring_sqe* s = &sqes[idx];
        memset(s, 0, sizeof(*s));
        s->opcode = IORING_OP_READ;
        s->fd = file_fd;
        s->addr = (uint64_t)(uintptr_t)buf;
        s->len = len;
        s->off = offset;
        s->user_data = user_data;
        sq_array[idx] = idx;
        std::atomic_thread_fence(std::memory_order_release);
        __atomic_store_n(sq_tail, tail + 1, __ATOMIC_RELEASE);
        ++pending_sqes;
    }

    // submit the pending SQEs; wait for at least `min_complete` completions in the ring. Returns submitted or -errno.
    int submit(unsigned min_complete = 0) {
        const unsigned n = pending_sqes;
        if (n == 0 && min_complete == 0) return 0;
        for (;;) {
            const int r = sys_enter(fd, n, min_complete, min_complete ? IORING_ENTER_GETEVENTS : 0);
            if (r < 0) {
                if (errno == EINTR) continue;
                return -errno;
            }
            pending_sqes -= (unsigned)r;
            in_flight += (unsigned)r;
            return r;
        }
    }

    // wait until at least `min_complete` completions are available (none submitted)
    int wait(unsigned min_complete) {
        for (;;) {
            const int r = sys_enter(fd, 0, min_complete, IORING_ENTER_GETEVENTS);
            if (r < 0) {
                if (errno == EINTR) continue;
                return -errno;
            }
            return 0;
        }
    }

    // pop one completion; returns false when the CQ is empty
    bool reap(uint64_t* user_data, int* res) {
        const unsigned head = *cq_head;
        const unsigned tail = __atomic_load_n(cq_tail, __ATOMIC_ACQUIRE);
        if (head == tail) return false;
        const io_uring_cqe* c = &cqes[head & *cq_mask];
        *user_data = c->user_data;
        *res = c->res;
        __atomic_store_n(cq_head, head + 1, __ATOMIC_RELEASE);
        --in_flight;
        return true;
    }

    unsigned cq_ready() const { return __atomic_load_n(cq_tail, __ATOMIC_ACQUIRE) - *cq_head; }
};

static inline void* alloc_aligned(size_t bytes, size_t align = 4096) {
    void* p = nullptr;
    if (posix_memalign(&p, align, bytes) != 0) return nullptr;
    return p;
}

}  // namespace uring
