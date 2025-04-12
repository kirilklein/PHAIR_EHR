# === General Configuration Keys ===
DELAYS = "delays"
CRITERIA_DEFINITIONS = "criteria_definitions"
CRITERIA = "criteria"
TIME_WINDOW_DAYS = "time_window_days"

# === Criteria Config Keys (used in YAML/JSON configuration) ===
CODE_ENTRY = "codes"
EXCLUDE_CODES = "exclude_codes"
CODE_GROUPS = "code_groups"
OPERATOR = "operator"
DAYS = "days"
EXPRESSION = "expression"

INCLUSION = "inclusion"
EXCLUSION = "exclusion"

MAX_COUNT = "max_count"

# Value/Limit Keys for Criteria
MIN_AGE = "min_age"
MAX_AGE = "max_age"
MIN_VALUE = "min_value"
MAX_VALUE = "max_value"
MIN_TIME = "min_time"
MAX_TIME = "max_time"
UNIQUE_CODE_LIMITS = "unique_code_limits"

# === Vectorized Extraction / Processing Keys ===
TIME_MASK = "time_mask"
CODE_MASK = "code_mask"
FINAL_MASK = "final_mask"
CRITERION_FLAG = "criterion_flag"
NUMERIC_VALUE = "numeric_value"
NUMERIC_VALUE_SUFFIX = "_numeric_value"
DELAY = "delay"
INDEX_DATE = "index_date"
AGE_AT_INDEX_DATE = "age_at_index_date"

# === Statistics ===
INITIAL_TOTAL = "initial_total"
EXCLUDED_BY_INCLUSION_CRITERIA = "excluded_by_inclusion_criteria"
EXCLUDED_BY_EXCLUSION_CRITERIA = "excluded_by_exclusion_criteria"
N_EXCLUDED_BY_EXPRESSION = "n_excluded_by_expression"
N_EXCLUDED_BY_CODE_LIMITS = "n_excluded_by_code_limits"
FINAL_INCLUDED = "final_included"


ALLOWED_OPERATORS = {"|", "&", "~", "and", "or", "not", "(", ")"}
