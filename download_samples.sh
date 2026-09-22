#!/bin/bash

# ============================================================================
# Download the raw FASTQ files for the samples listed in a LAMPP task CSV.
# ============================================================================
#
# Usage:
#   bash download_samples.sh [OPTIONS] <task_csv> <output_dir>
#
# Arguments:
#   task_csv     A LAMPP task CSV (train, test or test_gt) with a 'sample_id' column
#   output_dir   Where the fastq files are written
#
# Options:
#   -j, --jobs N          Number of samples to download concurrently (default: 1)
#       --limit N         Only process the first N samples
#       --start N         Start from sample N (1-indexed), for resuming a partial run
#       --dm7-base-url U  Base URL for the dm7 test reads (see below)
#   -h, --help            Show this message
#
# Requirements: sra-tools (fasterq-dump, prefetch), entrez-direct (esearch, efetch),
# and curl or wget, all on PATH. They are not part of the conda environments shipped
# with this repository; install them with e.g.
#   conda install -c bioconda -c conda-forge sra-tools entrez-direct
#
# Samples already present in the output directory are skipped, so an interrupted run can
# simply be restarted. A sample_id holding several SRA accessions joined by semicolons is
# downloaded run by run and concatenated. Single-end vs paired-end is detected
# automatically, and the output is named after the sample_id.
#
# Each accession is attempted with four methods in turn: fasterq-dump; prefetch +
# fasterq-dump; esearch/efetch, to find the run accessions behind a sample accession; and
# finally ENA's HTTPS mirror, which bypasses SRA-tools/VDB entirely.
#
# The dm7 test reads are an exception: they are anonymized samples that were never
# deposited in SRA, published instead as single-end, already-gzipped fastq files named
# after the anonymized sample id, and downloaded directly from ${DM7_BASE_URL}.
# ============================================================================

set -euo pipefail

# Where the dm7 test reads are published. Override with --dm7-base-url or by exporting
# DM7_BASE_URL.
DM7_BASE_URL="${DM7_BASE_URL:-https://files.cs.huji.ac.il/morani/LAMPP/DM7_test}"

# NCBI's E-utilities rate-limit aggressively (HTTP 429) and time out under load, so every
# network call goes through retry(). Export NCBI_API_KEY to raise the per-IP rate limit
# from 3 to 10 requests/s; the NCBI tools pick it up from the environment automatically.
NET_ATTEMPTS="${NET_ATTEMPTS:-5}"
NET_BACKOFF="${NET_BACKOFF:-15}"   # seconds; doubles on each attempt

# --split-3 rather than --split-files: on a run that yields three files, --split-files
# writes _1/_2/_3 and the _3 mate is silently dropped downstream. --split-3 writes _1/_2
# for the properly paired reads plus a bare <accession>.fastq for the orphans, which
# fold_orphans() then merges into R1.
FASTERQ_OPTS=(-e 1 --skip-technical --split-3)

have() { command -v "$1" >/dev/null 2>&1; }
log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

if have pigz; then GZIP=(pigz -c); GUNZIP=(pigz -dc)
else               GZIP=(gzip -c); GUNZIP=(gzip -dc); fi

# --- Arguments ---------------------------------------------------------------

print_usage() { sed -n '/^# Usage:/,/^# ===/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; }

JOBS=1
LIMIT=""
START=1
POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -j|--jobs)       JOBS="$2"; shift 2 ;;
        --limit)         LIMIT="$2"; shift 2 ;;
        --start)         START="$2"; shift 2 ;;
        --dm7-base-url)  DM7_BASE_URL="$2"; shift 2 ;;
        -h|--help)       print_usage; exit 0 ;;
        -*)              echo "Unknown option: $1" >&2; print_usage >&2; exit 1 ;;
        *)               POSITIONAL+=("$1"); shift ;;
    esac
done
set -- "${POSITIONAL[@]+"${POSITIONAL[@]}"}"

if [[ $# -ne 2 ]]; then
    echo "ERROR: expected a task CSV and an output directory" >&2
    print_usage >&2
    exit 1
fi

TASK_CSV="$1"
OUTPUT_DIR="$2"
DM7_BASE_URL="${DM7_BASE_URL%/}"

[[ -f "${TASK_CSV}" ]] || { echo "ERROR: CSV file not found: ${TASK_CSV}" >&2; exit 1; }
[[ "${JOBS}"  =~ ^[0-9]+$ && "${JOBS}"  -ge 1 ]] || { echo "ERROR: --jobs must be a positive integer" >&2; exit 1; }
[[ "${START}" =~ ^[0-9]+$ && "${START}" -ge 1 ]] || { echo "ERROR: --start must be a positive integer" >&2; exit 1; }
[[ -z "${LIMIT}" || "${LIMIT}" =~ ^[0-9]+$ ]]     || { echo "ERROR: --limit must be a non-negative integer" >&2; exit 1; }

mkdir -p "${OUTPUT_DIR}"

# One scratch directory for the whole run, holding a subdirectory per sample, so that
# concurrent workers cannot collide and a single trap cleans up after all of them.
SCRATCH_DIR="${OUTPUT_DIR}/.tmp_$$"
mkdir -p "${SCRATCH_DIR}"
trap 'rm -rf "${SCRATCH_DIR}"' EXIT

# --- Download methods --------------------------------------------------------

# Run a command, retrying on failure with exponential backoff and jitter. The jitter
# matters whenever --jobs > 1: without it, workers that all hit a 429 at the same moment
# would retry in lockstep and 429 again.
retry() {
    local what=$1; shift
    [[ "${1:-}" == "--" ]] && shift

    local attempt=1 delay="${NET_BACKOFF}" rc=0 wait_for
    while :; do
        "$@" && return 0 || rc=$?
        if [[ ${attempt} -ge ${NET_ATTEMPTS} ]]; then
            log "  ${what}: failed after ${attempt} attempt(s) (rc=${rc}); giving up"
            return ${rc}
        fi
        wait_for=$(( delay + RANDOM % (delay / 2 + 1) ))
        log "  ${what}: attempt ${attempt} failed (rc=${rc}); retrying in ${wait_for}s"
        sleep "${wait_for}"
        attempt=$(( attempt + 1 ))
        delay=$(( delay * 2 ))
    done
}

# --split-3 writes orphan reads to a bare <accession>.fastq next to _1/_2. Fold them into
# R1 so nothing is lost and so a bare <accession>.fastq unambiguously means "single-end",
# never "paired plus leftovers" -- which is what the layout check in assemble() relies on.
fold_orphans() {
    local acc=$1 dir=$2
    if [[ -f "${dir}/${acc}_1.fastq" && -f "${dir}/${acc}.fastq" ]]; then
        cat "${dir}/${acc}.fastq" >> "${dir}/${acc}_1.fastq"
        rm -f "${dir}/${acc}.fastq"
    fi
}

# A failed fasterq-dump can leave a partial fastq behind; clear it so the next attempt
# (or the caller's success check) is not fooled by the debris.
try_fasterq() {
    local acc=$1 dir=$2
    fasterq-dump "${acc}" -O "${dir}" -t "${dir}" "${FASTERQ_OPTS[@]}" 2>&1 && return 0
    rm -f "${dir}/${acc}"*.fastq
    return 1
}

method_fasterq() {
    local acc=$1 dir=$2
    have fasterq-dump || { log "fasterq-dump is not on PATH"; return 1; }
    retry "fasterq-dump ${acc}" -- try_fasterq "${acc}" "${dir}" || return 1
    fold_orphans "${acc}" "${dir}"
}

try_prefetch() {
    local acc=$1 pdir=$2
    # --max-size 50G: the 20G default silently refuses the largest runs.
    prefetch --max-size 50G "${acc}" -O "${pdir}" 2>&1 && return 0
    rm -rf "${pdir:?}/${acc}"
    return 1
}

method_prefetch() {
    local acc=$1 dir=$2
    local pdir="${dir}/prefetch_${acc}" rc=1
    have prefetch || { log "prefetch is not on PATH"; return 1; }

    mkdir -p "${pdir}"
    if retry "prefetch ${acc}" -- try_prefetch "${acc}" "${pdir}" &&
       retry "fasterq-dump(sra) ${acc}" -- fasterq-dump "${pdir}/${acc}/${acc}.sra" \
                -O "${dir}" -t "${dir}" "${FASTERQ_OPTS[@]}"; then
        fold_orphans "${acc}" "${dir}"
        rc=0
    else
        rm -f "${dir}/${acc}"*.fastq
    fi
    rm -rf "${pdir}"
    return ${rc}
}

# Fetch the runinfo table for an accession. One attempt; retry() wraps it.
#
# esearch failing (a 429, a curl timeout, DNS) has to propagate. Piping esearch straight
# into the parser instead would make a rate-limited lookup look exactly like a genuine
# "this accession has no runs" answer, and the sample would fail instead of being retried.
try_runinfo() {
    local acc=$1 out=$2
    esearch -db sra -query "${acc}" | efetch -format runinfo > "${out}" 2>/dev/null
    local pipe=("${PIPESTATUS[@]}")
    [[ "${pipe[0]}" -eq 0 && "${pipe[1]}" -eq 0 ]] || return 1
    grep -q '^Run,' "${out}"   # a valid runinfo table always has this header
}

method_esearch() {
    local acc=$1 dir=$2
    local runinfo="${dir}/${acc}.runinfo.csv" alts alt file
    # Entrez Direct is optional; bail out at once rather than burning the whole retry
    # budget on "command not found" before reaching the ENA fallback.
    have esearch && have efetch || { log "esearch/efetch are not on PATH"; return 1; }

    retry "esearch ${acc}" -- try_runinfo "${acc}" "${runinfo}" || { rm -f "${runinfo}"; return 1; }
    alts=$(tail -n +2 "${runinfo}" | cut -d',' -f1 | grep -E '^[SED]RR' || true)
    rm -f "${runinfo}"
    [[ -n "${alts}" ]] || { log "no alternative accessions found for ${acc}"; return 1; }

    for alt in ${alts}; do
        log "trying alternative accession ${alt}"
        if method_fasterq "${alt}" "${dir}"; then
            for file in "${dir}/${alt}"*.fastq; do
                [[ -f "${file}" ]] && mv "${file}" "${file//${alt}/${acc}}"
            done
            return 0
        fi
    done
    return 1
}

# Stream-decompress one ENA fastq.gz URL into a plain .fastq, matching what the other
# methods produce so that assemble() is indifferent to which one fetched the data.
#
# The subshell's `set -o pipefail` is required: without it, a failed curl followed by gzip
# hitting a clean EOF on empty stdin would report the pipeline as successful.
try_ena_file() {
    local url=$1 out=$2
    ( set -o pipefail; curl -fsSL --max-time 3600 "https://${url}" | "${GUNZIP[@]}" ) > "${out}"
    local rc=$?
    [[ ${rc} -ne 0 ]] && rm -f "${out}"
    return ${rc}
}

# Last-resort fallback: fetch the fastq files directly from ENA's HTTPS mirror. Some runs
# (we hit this on several GHS training samples) fail fasterq-dump reproducibly -- including
# against a prefetch-verified local .sra file, which rules out a network cause -- because
# sra-tools cannot read runs NCBI has migrated to SRA-Lite-only storage. ENA mirrors the
# same runs as plain fastq.gz with no VDB involved.
method_ena() {
    local acc=$1 dir=$2
    local field urls=() i out

    field=$(curl -fsSL --max-time 30 "https://www.ebi.ac.uk/ena/portal/api/filereport\
?accession=${acc}&result=read_run&fields=fastq_ftp&format=tsv" 2>/dev/null | tail -n +2 | cut -f2)
    [[ -n "${field}" ]] || { log "no fastq_ftp entry at ENA for ${acc}"; return 1; }

    IFS=';' read -r -a urls <<< "${field}"
    log "ENA: found ${#urls[@]} file(s) for ${acc}"

    for i in "${!urls[@]}"; do
        # A single file is a single-end run; two or more are the mates of a paired one.
        [[ ${#urls[@]} -eq 1 ]] && out="${dir}/${acc}.fastq" || out="${dir}/${acc}_$(( i + 1 )).fastq"
        if ! retry "ENA ${out##*/}" -- try_ena_file "${urls[$i]}" "${out}"; then
            # Never leave a lone R1 behind: it would be taken for a single-end run.
            rm -f "${dir}/${acc}"*.fastq
            return 1
        fi
    done
}

download_accession() {
    local acc=$1 dir=$2 method
    for method in fasterq prefetch esearch ena; do
        log "Downloading ${acc} via ${method}..."
        "method_${method}" "${acc}" "${dir}" && return 0
    done
    log "ERROR: all download methods failed for ${acc}"
    return 1
}

fetch_url() {
    local url=$1 out=$2
    if have wget; then wget -q -O "${out}" "${url}"
    else               curl -fsSL --max-time 3600 -o "${out}" "${url}"; fi && return 0
    rm -f "${out}"
    return 1
}

method_dm7() {
    local sample_id=$1
    local url="${DM7_BASE_URL}/${sample_id}.fastq.gz"
    local part="${OUTPUT_DIR}/${sample_id}.fastq.gz.part"

    log "Downloading ${sample_id} from ${url}"
    # Write to .part and rename only on success, so an interrupted run leaves nothing that
    # the skip check would accept as a finished sample.
    if retry "dm7 ${sample_id}" -- fetch_url "${url}" "${part}" && [[ -s "${part}" ]]; then
        mv "${part}" "${OUTPUT_DIR}/${sample_id}.fastq.gz"
        log "Created: ${OUTPUT_DIR}/${sample_id}.fastq.gz"
        return 0
    fi
    rm -f "${part}"
    log "ERROR: could not download ${url}"
    return 1
}

# --- Assembling and driving --------------------------------------------------

# Write the downloaded accessions out as gzipped fastq named after the sample_id. Several
# accessions are concatenated in order; `cat | gzip` streams straight to the destination,
# so no intermediate combined fastq is ever written.
assemble() {
    local sample_id=$1 temp_dir=$2; shift 2
    local accessions=("$@")
    local suffixes=("") produced=() suffix acc srcs dst

    # fold_orphans() guarantees that a bare <acc>.fastq means a single-end run.
    [[ -f "${temp_dir}/${accessions[0]}_1.fastq" ]] && suffixes=("_1" "_2")

    for suffix in "${suffixes[@]}"; do
        srcs=()
        for acc in "${accessions[@]}"; do
            if [[ -f "${temp_dir}/${acc}${suffix}.fastq" ]]; then
                srcs+=("${temp_dir}/${acc}${suffix}.fastq")
            else
                log "WARNING: expected file not found: ${acc}${suffix}.fastq"
            fi
        done
        dst="${OUTPUT_DIR}/${sample_id}${suffix}.fastq.gz"
        # The exit status has to be checked explicitly: this runs inside an `if`, which
        # disables `set -e` for its whole dynamic extent, so a gzip that died partway
        # would otherwise leave a truncated .gz behind and still be reported a success.
        if [[ ${#srcs[@]} -eq 0 ]] || ! cat "${srcs[@]}" | "${GZIP[@]}" > "${dst}" || [[ ! -s "${dst}" ]]; then
            log "ERROR: could not write ${dst##*/}"
            # Never leave a lone R1 behind: it would be taken for a single-end sample.
            rm -f "${dst}" ${produced[@]+"${produced[@]}"}
            return 1
        fi
        produced+=("${dst}")
        log "Created: ${dst}"
    done
}

process_sample() {
    local sample_id=$1
    local temp_dir="${SCRATCH_DIR}/$2"

    log "Processing sample: ${sample_id}"
    [[ "${sample_id}" == dm7_test_* ]] && { method_dm7 "${sample_id}"; return $?; }

    mkdir -p "${temp_dir}"

    # A sample backed by several SRA runs carries them semicolon-joined in its sample_id.
    local accessions=() i
    IFS=';' read -ra accessions <<< "${sample_id}"
    for i in "${!accessions[@]}"; do
        accessions[i]=$(echo "${accessions[i]}" | xargs)   # trim whitespace
        download_accession "${accessions[i]}" "${temp_dir}" || return 1
    done

    assemble "${sample_id}" "${temp_dir}" "${accessions[@]}"
}

# A sample is done if any of its possible outputs is present. Both the gzipped and the
# uncompressed names are checked: the script always gzips before it finishes, so checking
# only the uncompressed names would re-download everything on a restart.
sample_is_done() {
    local s="${OUTPUT_DIR}/$1"
    [[ -f "${s}.fastq" || -f "${s}.fastq.gz" || -f "${s}_1.fastq" || -f "${s}_1.fastq.gz" ]]
}

# Each worker records its outcome in a file named after the sample's index, so the parent
# can tally the results without sharing state with background subshells.
run_one() {
    local sample_id=$1 idx=$2
    if sample_is_done "${sample_id}"; then
        log "Sample ${sample_id} already exists in the output directory. Skipping."
        echo "skipped" > "${STATUS_DIR}/${idx}"
    elif process_sample "${sample_id}" "${idx}"; then
        log "COMPLETED: ${sample_id}"
        echo "ok" > "${STATUS_DIR}/${idx}"
    else
        log "FAILED: ${sample_id}"
        echo "failed ${sample_id}" > "${STATUS_DIR}/${idx}"
    fi
    rm -rf "${SCRATCH_DIR:?}/${idx}"
    return 0
}

# --- Main --------------------------------------------------------------------

log "=========================================="
log "Starting downloads"
log "CSV file:         ${TASK_CSV}"
log "Output directory: ${OUTPUT_DIR}"
log "Concurrent jobs:  ${JOBS}"
log "=========================================="

# Read every sample_id out of the CSV. The header is searched for the column literally
# named 'sample_id', which is column 1 in the test CSVs and column 2 in the train CSVs.
mapfile -t SAMPLE_IDS < <(
    awk -F',' 'NR==1 {for (i = 1; i <= NF; i++) if ($i == "sample_id") col = i
                      if (!col) exit 1
                      next}
               {gsub(/^[ \t]+|[ \t\r]+$/, "", $col); if ($col != "") print $col}' "${TASK_CSV}"
)

if [[ ${#SAMPLE_IDS[@]} -eq 0 ]]; then
    echo "ERROR: no sample ids found in ${TASK_CSV} (is there a 'sample_id' column?)" >&2
    exit 1
fi
log "Found ${#SAMPLE_IDS[@]} sample(s) in the CSV file"

if [[ $(( START - 1 )) -ge ${#SAMPLE_IDS[@]} ]]; then
    echo "ERROR: --start ${START} is past the end of the CSV (${#SAMPLE_IDS[@]} samples)" >&2
    exit 1
fi
SAMPLE_IDS=("${SAMPLE_IDS[@]:$(( START - 1 ))}")
[[ -n "${LIMIT}" && ${LIMIT} -lt ${#SAMPLE_IDS[@]} ]] && SAMPLE_IDS=("${SAMPLE_IDS[@]:0:${LIMIT}}")
log "Processing ${#SAMPLE_IDS[@]} sample(s), starting from sample ${START}"

STATUS_DIR="${SCRATCH_DIR}/status"
mkdir -p "${STATUS_DIR}"

for idx in "${!SAMPLE_IDS[@]}"; do
    if [[ ${JOBS} -eq 1 ]]; then
        run_one "${SAMPLE_IDS[${idx}]}" "${idx}"
    else
        # Keep at most ${JOBS} workers alive at a time.
        while [[ $(jobs -rp | wc -l) -ge ${JOBS} ]]; do wait -n || true; done
        run_one "${SAMPLE_IDS[${idx}]}" "${idx}" &
    fi
done
wait

N_OK=0
N_SKIPPED=0
FAILED_SAMPLES=()
for idx in "${!SAMPLE_IDS[@]}"; do
    status=$(cat "${STATUS_DIR}/${idx}" 2>/dev/null || echo "failed ${SAMPLE_IDS[${idx}]}")
    case "${status}" in
        ok)      N_OK=$(( N_OK + 1 )) ;;
        skipped) N_SKIPPED=$(( N_SKIPPED + 1 )) ;;
        *)       FAILED_SAMPLES+=("${status#failed }") ;;
    esac
done

log "=========================================="
log "Downloaded: ${N_OK}   Already present: ${N_SKIPPED}   Failed: ${#FAILED_SAMPLES[@]}"
if [[ ${#FAILED_SAMPLES[@]} -gt 0 ]]; then
    log "Failed samples:"
    printf '    %s\n' "${FAILED_SAMPLES[@]}"
    exit 1
fi
log "=========================================="
