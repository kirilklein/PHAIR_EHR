# Oracle Feature Definitions

All features are extracted from the pre-index patient history by `corebehrt/modules/simulation/oracle_features.py`.

Features are optionally z-scored (mean=0, std=1) when `features.standardize: true`, so each coefficient represents the effect of a 1-SD change.

## Feature Table

| Feature | Type | Description | Window |
|---------|------|-------------|--------|
| `recent_event_count` | count | Number of events (any code type) before index | `recent_window_days` (default: 90d) |
| `disease_burden` | unique count | Unique diagnosis codes before index | `lookback_days` (default: 365d) |
| `medication_count` | unique count | Unique medication codes before index (polypharmacy proxy) | `lookback_days` |
| `utilization_intensity` | count | Total events before index | `lookback_days` |
| `age` | continuous | Age in years at index date (from DOB events) | full history |
| `chronic_disease_count` | unique count | Distinct diagnosis code groups (first 5 chars) | full history |
| `code_diversity` | unique count | Total unique codes across full history | full history |
| `event_recency` | days | Days since most recent event before index | full history |
| `recent_burst_ratio` | ratio | events_in_burst_window / (events_in_lookback + 1) | `burst_window_days` / `lookback_days` |
| `sequence_motif_count` | count | (diagnosis -> medication) pairs within motif window | `motif_window_days` (default: 30d) |

## Code Prefix Configuration

Features that filter by medical concept type use configurable prefixes:

```yaml
features:
  code_prefixes:
    diagnosis: "D/"       # Used by: disease_burden, chronic_disease_count, sequence_motif_count
    medication: "M/"      # Used by: medication_count, sequence_motif_count
    procedure: "P/"       # Available but not currently used by any feature
    admission: "ADM/"     # Available but not currently used by any feature
```

## Notes

- If no codes match a prefix (e.g., no medication codes in the data), the feature is filled with 0 and a warning is logged.
- `age` requires DOB events in the history. Missing DOB is filled with mean age.
- `event_recency` for patients with no events is filled with mean recency.
- `chronic_disease_count` uses the first 5 characters of diagnosis codes as group keys. The granularity depends on the coding system.
- `sequence_motif_count` counts (diagnosis, medication) pairs where the medication occurs within `motif_window_days` after the diagnosis.
