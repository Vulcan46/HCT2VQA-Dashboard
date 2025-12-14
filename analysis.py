import json
import pandas as pd
import os
import glob

# 1. Configuration
# Map your specific JSON keys to the 4 main buckets
CATEGORY_MAP = {
    'Subject_Consistency': 'Subject',
    'Action_Consistency': 'Action',
    'Env_Consistency': 'Environment',
    'Audio_Consistency': 'Audio'
}

# Full names for cleaner output
PROMPT_CAT_NAMES = {
    'bi': 'Biological Implausibility',
    'phy': 'Physical Incongruity',
    'si': 'Social Inversion',
    'tm': 'Temporal Modification'
}


def load_and_process_data(file_list):
    all_data = []

    for filepath in file_list:
        # Extract metadata from filename
        filename = os.path.basename(filepath)
        name_parts = filename.split('_')
        prompt_cat_code = name_parts[0]  # bi, phy, si, tm
        model_name = name_parts[1].split('.')[0]  # sora2, veo3

        # UTF-8 Encoding Fix included
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for entry in data:
            prompt_id = entry['prompt_id']
            full_category = entry.get('prompt_category', prompt_cat_code)

            # Iterate through the 4 question types
            for json_key, clean_cat in CATEGORY_MAP.items():
                if json_key in entry['evaluation_questions']:
                    for q in entry['evaluation_questions'][json_key]:
                        # Normalize Answer
                        ans_raw = q['answer']
                        score = None
                        if ans_raw:
                            score = 1 if ans_raw.lower() == 'yes' else 0

                        all_data.append({
                            'Model': model_name,
                            'Prompt_Category_Code': prompt_cat_code,
                            'Prompt_Category_Name': PROMPT_CAT_NAMES.get(prompt_cat_code, prompt_cat_code),
                            'Prompt_ID': prompt_id,
                            'Question_Type': clean_cat,
                            'Question_ID': q['question_id'],
                            'Score': score
                        })

    return pd.DataFrame(all_data)


# --- Analysis Functions ---

def analyze_performance(df):
    # Drop N/A values for calculation
    clean_df = df.dropna(subset=['Score'])

    print("--- 1. OVERALL WINNER (Global Alignment) ---")
    print(clean_df.groupby('Model')['Score'].mean().apply(lambda x: f"{x:.1%}"))
    print("\n")

    print("--- 2. CATEGORY BREAKDOWN (Where do they fail?) ---")
    cat_breakdown = clean_df.groupby(['Model', 'Prompt_Category_Name'])['Score'].mean().unstack()
    print(cat_breakdown.map(lambda x: f"{x:.1%}"))
    print("\n")

    print("--- 3. QUESTION TYPE CAPABILITIES (Static vs. Dynamic) ---")
    q_breakdown = clean_df.groupby(['Model', 'Question_Type'])['Score'].mean().unstack()
    print(q_breakdown.map(lambda x: f"{x:.1%}"))
    print("\n")

    print("--- 4. THE 'PRIOR BIAS' DETECTOR (Action vs Subject Gap) ---")
    for model in clean_df['Model'].unique():
        m_df = clean_df[clean_df['Model'] == model]
        subj_score = m_df[m_df['Question_Type'] == 'Subject']['Score'].mean()
        action_score = m_df[m_df['Question_Type'] == 'Action']['Score'].mean()
        gap = subj_score - action_score
        print(f"{model}: Subject ({subj_score:.1%}) - Action ({action_score:.1%}) = Drop-off of {gap:.1%}")
    print("\n")

    print("--- 5. AUDIO-VISUAL DISCONNECT ---")
    for model in clean_df['Model'].unique():
        m_df = clean_df[clean_df['Model'] == model]
        visual_score = m_df[m_df['Question_Type'].isin(['Subject', 'Action', 'Environment'])]['Score'].mean()
        audio_score = m_df[m_df['Question_Type'] == 'Audio']['Score'].mean()
        print(f"{model}: Visual {visual_score:.1%} vs Audio {audio_score:.1%}")
    print("\n")

    # --- NEW SECTION ---
    print("--- 6. DEEP DIVE: Question Alignment by Prompt Category ---")
    print("(Shows how each Question Type performs within each specific Category)")

    # Group by Model, Prompt Category, AND Question Type
    deep_dive = clean_df.groupby(['Model', 'Prompt_Category_Name', 'Question_Type'])['Score'].mean().unstack()

    for model in clean_df['Model'].unique():
        print(f"\n[ Model: {model} ]")
        # Select data for this model
        model_data = deep_dive.loc[model]
        # Reorder columns for logical flow
        cols = ['Subject', 'Environment', 'Action', 'Audio']
        model_data = model_data[cols]
        print(model_data.map(lambda x: f"{x:.1%}"))


# --- Execution ---
if __name__ == "__main__":
    directory_path = "C://Users/advai/OneDrive/Documents/AdvancedAI/Data"
    # Pattern to match all files ending with '.json'
    pattern = os.path.join(directory_path, '*.json')
    # Point this to where your .json files are
    files = glob.glob(pattern)
    if files:
        df = load_and_process_data(files)
        analyze_performance(df)
    else:
        print("No JSON files found in the current directory.")