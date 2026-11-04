#!/usr/bin/env bash
# E92 safety watcher: every 20 s, check the machine; on any change kill every E92 job and leave a STOP file.
#   setsid nohup tools/e92_watch.sh > /home/data/e92_scratch/watch.log 2>&1 &
stop=/home/data/e92_scratch/STOP
while true; do
  offline="$(cat /sys/devices/system/cpu/offline)"
  mce="$(journalctl -k -b 0 | grep -cE 'MC[0-9]+_STATUS|Machine Check: .*Bank [0-9]+' || true)"
  if [[ "$offline" != "10,26" || "$mce" != "1" ]]; then
    echo "$(date -u +%FT%TZ) STOP: offline=$offline machine_checks=$mce" | tee -a "$stop"
    pkill -f 'e92_(full|audit)_driver.sh' || true
    pkill -f 'tools/run_e92_' || true
    exit 1
  fi
  if [[ -f /home/data/e92_scratch/WATCH_EXIT ]]; then echo "$(date -u +%FT%TZ) exit requested"; exit 0; fi
  sleep 20
done
