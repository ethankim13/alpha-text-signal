'''
1. Initializes the DB (picks up filing_embeddings if not already there)
2. Finds sections still missing an embedding
3. For each: generate the embedding, write it to filing_embeddings
4. Prints a summary at the end
'''
import sys 
from pathlib import Path
# Let this script find config.py and src/ when run directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import embedder, db_utils
from tqdm import tqdm

import config

def main():
    print("Initializing database...")
    db_utils.init_db()

    sections = db_utils.get_sections_needing_embedding()
    # sections = sections[:10] - test on small batch
    print(f"Found {len(sections)} sections needing embedding.\n")

    embedded_count = 0
    skipped_count = 0

    for section in tqdm(sections, desc = "Sections"): # tqdm adds live progress bar
        embedding = embedder.embed_text(section["section_text"]) # save result

        if embedding is not None:
            db_utils.insert_filing_embedding(section["accession_number"],
                                             section["section_type"],
                                             embedding)
            embedded_count += 1
        else:
            skipped_count += 1

    print("\nDone.")
    print(f"Total found: {embedded_count + skipped_count}")
    print(f"Amount embedded: {embedded_count}")
    print(f"Amount skipped: {skipped_count}")
    
if __name__ == "__main__":
    main()
        

