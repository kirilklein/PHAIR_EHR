from corebehrt.main_causal.helper.build_tree import build_tree, extract_dict_at_level
import pickle


def main():
    csv_file = (
        "corebehrt/main_causal/helper/data/diag.csv"  # Adjust the path as needed.
    )

    # Build the tree.
    tree = build_tree(csv_file)

    # Extract the dictionary for the desired grouping level.
    # For example, with this scheme:
    #   Level 1: Chapters (e.g. "Chap. I: ...")
    #   Level 2: Subchapters (e.g. "Infectious intestinal diseases [DA00-DA09]")
    #   Level 3: Detailed diagnosis groups (e.g. "Cholera", "Typhoid and paratyphoid", "Other salmonella infections")
    target_level = 3
    outcome_dict = extract_dict_at_level(tree, target_level)

    print("\nOutcome Dictionary at level", target_level)
    for disease, codes in list(outcome_dict.items())[:5]:
        print("Disease:", disease)
        print("Codes:", codes)
        print("-" * 40)

    with open("corebehrt/main_causal/helper/data/diag_level_3.pkl", "wb") as f:
        pickle.dump(outcome_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
