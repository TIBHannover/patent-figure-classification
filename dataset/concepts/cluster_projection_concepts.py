import re
import json
import nltk

def preprocess_concept(concept):
    """Preprocess concept label"""
    concept = concept.encode('utf-8').decode('unicode_escape')
    concept = re.sub(r'\d+', '', concept)
    concept = re.sub(r'[^\w\s]', '', concept.lower()).strip()
    concept = concept.replace(" view", "")
    return concept

with open("projection_concepts.json", "r") as rf:
    """Get list of all projection concepts"""
    projection_concepts = json.load(rf)["concepts"]
    concepts = [preprocess_concept(concept) for concept in projection_concepts]

projection_hierarchy = {
    "projections": {
        "perspective": ["perspective"],
        "oblique": ["oblique", "cabinet", "cavalier", "military"],
        "axonometric": ["axonometric", "isometric", "dimetric", "trimetric"],
        "sectional": ["sectional", "cross-section", "cross section", "crosssectional"],
        "detail": ["detail", "enlarged", "exploded", "magnify", "expand",
                   "close up", "zoom", "closeup", "zoomedin"],
        "plan": ["plan", "top", "bottom", "underside", "topside"],
        "elevational": [
            "elevational", "front", "back", "left", "right",
            "side", "rear", "heel", "face", "end", "rearside",
            "frontend", "backend", "leftside", "rightside"]
    }
}

stemmer = nltk.stem.PorterStemmer()

projection2keywords = {
    projection: [stemmer.stem(keyword) for keyword in keywords]
    for projection, keywords in projection_hierarchy["projections"].items()
}

def classify_concept(concept):
    """Map the concept label text to a standardized concept label"""
    concept_tokens = [stemmer.stem(token) for token in concept.split(" ") if token]

    for concept_token in concept_tokens[::-1]:
        for projection in projection2keywords.keys():
            if concept_token in projection2keywords[projection]:
                return projection
    
concept2projection = {}
projection2concept = {}

print(f"Total concepts: {len(concepts)}")

for concept in concepts:
    projection = classify_concept(concept)
    concept2projection[concept] = projection
    projection2concept.setdefault(projection, set())
    projection2concept[projection].add(concept)

for projection in projection2concept.keys():
    projection2concept[projection] = list(projection2concept[projection])
    print(f"{projection}: {len(projection2concept[projection])}")

## Save the mappings ##
with open("concept2projection.json", "w") as wf:
    json.dump(concept2projection, wf, indent=4)

with open("projection2concept.json", "w") as wf:
    json.dump(projection2concept, wf, indent=4)



