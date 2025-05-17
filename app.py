from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import uvicorn
from typing import List
from rapidfuzz.distance.JaroWinkler import normalized_similarity
from itertools import permutations
import re

app = FastAPI()

def read_file(file: UploadFile) -> pd.DataFrame:
    try:
        if file.filename.endswith('.csv'):
            return pd.read_csv(file.file)
        elif file.filename.endswith('.xlsx'):
            return pd.read_excel(file.file)
        else:
            raise ValueError("Unsupported file type. Only .csv and .xlsx are allowed.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read {file.filename}: {str(e)}")

@app.post("/upload")
async def upload_files(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...)
) -> List[dict]:
    try:
        master_df = read_file(file1)
        test_df = read_file(file2)
        
        # Validate required columns
        if 'Master_Code' not in master_df.columns or 'Master_Name' not in master_df.columns:
            raise HTTPException(status_code=400, detail="Master dataset must contain 'Master_Code' and 'Master_Name' columns.")
        elif 'Buyer_Name' not in test_df.columns:
            raise HTTPException(status_code=400, detail="Test dataset must contain 'Buyer_Name' column.")
        else:

            # Normalize names
            master_df['Master_Name_Clean'] = master_df['Master_Name'].str.lower().apply(str.strip)
            test_df['Buyer_Name_Clean'] = test_df['Buyer_Name'].str.lower().apply(str.strip)

            similar_match=[]

            for i, test_name_clean in enumerate(test_df['Buyer_Name_Clean']):
                buyer_name = test_df['Buyer_Name'].iloc[i]

                # Try exact match
                exact_match = master_df[master_df['Master_Name_Clean'] == test_name_clean]

                if not exact_match.empty:
                    continue
                
                else:
                    # Function to clean company suffixes (for Jaccard only)
                    def clean_company_name_for_jaccard(name):
                        suffixes = r"\b(incorporated|inc|llc|ltd|limited|corp|corporation|plc|co|company|pvt|private)\b"
                        name = name.lower()
                        name = re.sub(suffixes, '', name)
                        name = re.sub(r'\s+', ' ', name)  
                        return name.strip()

                    def permuted_winkler_distance(a, b):
                        """
                        Computes a distance = 1 - max Jaro-Winkler similarity over all token permutations of `a`.
                        Lower distance = better match.
                        """
                        tokens = a.split()
                        max_sim = 0.0
                        for perm in permutations(tokens):
                            permuted = " ".join(perm)
                            sim = normalized_similarity(permuted, b) / 100  # convert 0–100 to 0–1
                            if sim > max_sim:
                                max_sim = sim
                        return 1.0 - max_sim

                    def jaccard_distance(a, b):
                        """
                        Computes Jaccard distance after cleaning common company suffixes.
                        """
                        a_clean = clean_company_name_for_jaccard(a)
                        b_clean = clean_company_name_for_jaccard(b)

                        set_a = set(a_clean.split())
                        set_b = set(b_clean.split())
                        intersection = set_a & set_b
                        union = set_a | set_b
                        if not union:
                            return 1.0
                        return 1.0 - len(intersection) / len(union)

                    # Compute distances using both methods
                    winkler_distances = master_df['Master_Name_Clean'].apply(
                        lambda master_clean: permuted_winkler_distance(test_name_clean, master_clean)
                    )
                    jaccard_distances = master_df['Master_Name_Clean'].apply(
                        lambda master_clean: jaccard_distance(test_name_clean, master_clean)
                    )

                    # Get top 10 indices from each
                    top_winkler = winkler_distances.nsmallest(10).index.tolist()
                    top_jaccard = jaccard_distances.nsmallest(10).index.tolist()

                    # Interleave results
                    interleaved = []
                    for w, j in zip(top_winkler, top_jaccard):
                        interleaved.append(w)
                        interleaved.append(j)

                    # Remove duplicates while preserving order
                    unique_indices = list(dict.fromkeys(interleaved))

                    # Limit to top 10 and map to Master_Name
                    top_matches = master_df.loc[unique_indices[:10], 'Master_Name'].tolist()

                    similar_match.append({buyer_name:top_matches})

            return similar_match

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
