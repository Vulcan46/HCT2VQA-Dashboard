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


def load_and_process_data(file_list):
    all_data = []

    for filepath in file_list:
        # Extract metadata from filename (e.g., "bi_sora2.json" -> category="bi", model="sora2")
        filename = os.path.basename(filepath)
        name_parts = filename.split('_')
        prompt_cat_code = name_parts[0]  # bi, phy, si, tm
        model_name = name_parts[1].split('.')[0]  # sora2, veo3

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for entry in data:
            prompt_id = entry['prompt_id']
            full_category = entry.get('prompt_category', prompt_cat_code)  # Use internal category if available

            # Iterate through the 4 question types
            for json_key, clean_cat in CATEGORY_MAP.items():
                if json_key in entry['evaluation_questions']:
                    for q in entry['evaluation_questions'][json_key]:
                        # Normalize Answer: 1 for Yes, 0 for No, None for Null
                        ans_raw = q['answer']
                        score = None
                        if ans_raw:
                            score = 1 if ans_raw.lower() == 'yes' else 0

                        all_data.append({
                            'Model': model_name,
                            'Prompt_Category_Code': prompt_cat_code,
                            'Full_Category': full_category,
                            'Prompt_ID': prompt_id,
                            'Question_Type': clean_cat,  # Subject, Action, etc.
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
    # Pivot table: Model vs Prompt Category
    cat_breakdown = clean_df.groupby(['Model', 'Prompt_Category_Code'])['Score'].mean().unstack()
    print(cat_breakdown.applymap(lambda x: f"{x:.1%}"))
    print("\n")

    print("--- 3. QUESTION TYPE CAPABILITIES (Static vs. Dynamic) ---")
    # Pivot table: Model vs Question Type (Subject, Action, Audio, Env)
    q_breakdown = clean_df.groupby(['Model', 'Question_Type'])['Score'].mean().unstack()
    print(q_breakdown.applymap(lambda x: f"{x:.1%}"))
    print("\n")

    print("--- 4. THE 'PRIOR BIAS' DETECTOR (Action vs Subject Gap) ---")
    # Calculate how often Action fails relative to Subject
    # This specifically targets "Cat exists (Subject=1)" but "Cat Meows (Action=0)"
    for model in clean_df['Model'].unique():
        m_df = clean_df[clean_df['Model'] == model]
        subj_score = m_df[m_df['Question_Type'] == 'Subject']['Score'].mean()
        action_score = m_df[m_df['Question_Type'] == 'Action']['Score'].mean()
        gap = subj_score - action_score
        print(f"{model}: Subject ({subj_score:.1%}) - Action ({action_score:.1%}) = Drop-off of {gap:.1%}")
        if gap > 0.30:
            print(f"   -> WARNING: {model} shows strong Prior Bias (renders objects but ignores physics/logic).")
    print("\n")

    print("--- 5. AUDIO-VISUAL DISCONNECT ---")
    # Compare Visual Scores (Subj+Action+Env) vs Audio
    for model in clean_df['Model'].unique():
        m_df = clean_df[clean_df['Model'] == model]
        visual_score = m_df[m_df['Question_Type'].isin(['Subject', 'Action', 'Environment'])]['Score'].mean()
        audio_score = m_df[m_df['Question_Type'] == 'Audio']['Score'].mean()
        print(f"{model}: Visual {visual_score:.1%} vs Audio {audio_score:.1%}")

basepath = "C:/Users/advai/Downloads/answer_data-20251207T030202Z-3-001/answer_data/"
files = glob.glob(basepath + "*.json")
df = load_and_process_data(files)
analyze_performance(df)