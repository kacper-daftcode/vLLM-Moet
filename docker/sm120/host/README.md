# Host-side units of the rtxdev2 box (copies; the live files are under /etc)

- `gpu-power-cap.service` → `/etc/systemd/system/gpu-power-cap.service`, `gpu-power-cap.default` → `/etc/default/gpu-power-cap`:
  the GPUs' power cap at boot (`CAP_W`, 400 W since 2026-10-25: GPU 0 fell off the PCIe bus twice under four-GPU
  full-power prefill at 430–440 W per card; the RTX 5090's default is 550 W, max 600). Ordered after the older
  `nvidia-powerlimit.service` (persistence mode + the 550 W default, 2026-10-13) so that the cap always wins at boot.
  Change the number in `/etc/default/gpu-power-cap` and `systemctl restart gpu-power-cap`; `systemctl disable
  gpu-power-cap` restores the 550 W unit's behaviour at the next boot. `docker/sm120/run-dsv41-5090.sh` re-applies the cap
  before every boot of the hybrid when a card sits above it (`POWER_CAP_W`, 0 = skip).

Install: `cp gpu-power-cap.service /etc/systemd/system/ && cp gpu-power-cap.default /etc/default/gpu-power-cap &&
systemctl daemon-reload && systemctl enable --now gpu-power-cap`.
