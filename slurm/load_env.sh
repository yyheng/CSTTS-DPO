#!/bin/bash
# ---------------------------------------------------------------------------
# slurm/load_env.sh : load <repo>/config/paths.env into the calling shell WITHOUT overriding any
# CMI_DPO_* variable that is already set there (the same precedence as cmi_dpo/paths.py:
# environment wins over the file). Must be sourced, never executed:
#
#   source "$PKG/slurm/load_env.sh"      # PKG = the repository root; the file is <PKG>/config/paths.env
#
# The file itself is parsed by bash (`set -a; source`), so an `export ` prefix, quotes, $VARS and
# `# comments` behave exactly as in a plain `source`; afterwards every CMI_DPO_* variable that was
# set BEFORE the load is restored (and exported). A caller override such as
#   CMI_DPO_ENV_MAIN=other slurm/sb slurm/gen.sbatch
# therefore survives every re-load along the way (slurm/sb, the sbatch body, smoke.sh,
# run_dpo_round.sh). A missing file loads nothing, like paths.load_env_file(); the callers check
# for the file themselves. The repository root is derived from this file's own location.
# ---------------------------------------------------------------------------
cmi_dpo_load_env() {
  local file name allexport=0
  local -A keep=()
  file=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config/paths.env
  [[ -f "$file" ]] || return 0
  for name in "${!CMI_DPO_@}"; do keep[$name]=${!name}; done
  [[ $- == *a* ]] && allexport=1
  set -a
  # shellcheck disable=SC1090
  source "$file"
  (( allexport )) || set +a
  for name in "${!keep[@]}"; do export "$name=${keep[$name]}"; done
}
cmi_dpo_load_env
unset -f cmi_dpo_load_env
