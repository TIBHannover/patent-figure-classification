templates = {
    "T110": {
        "id": "T110",
        "question": "Is the figure of type '{type}'? Answer 'yes' or 'no'.",
        "type": "binary",
        "aspect": "type"
    },
    "T120": {
        "id": "T120",
        "question": "Is the figure shown in the projection '{projection}'? Answer 'yes' or 'no'.",
        "type": "binary",
        "aspect": "projection"
    },
    "T130": {
        "id": "T130",
        "question": "Is the figure showing the object '{object}'? Answer 'yes' or 'no'.",
        "type": "binary",
        "aspect": "object"
    },
    "T140": {
        "id": "T140",
        "question": "Is the figure from a patent in the USPC class '{uspc}'? Answer 'yes' or 'no'.",
        "type": "binary",
        "aspect": "uspc"
    },
    "T210": {
        "id": "T210",
        "question": "Which type does the figure belong to? Choose one correct option. Options: {options_text}.",
        "type": "multiple_choice",
        "aspect": "type",
    },
    "T220": {
        "id": "T220",
        "question": "Which projection is the figure shown in? Choose one correct option. Options: {options_text}.",
        "type": "multiple_choice",
        "aspect": "projection",
    },
    "T230": {
        "id": "T230",
        "question": "Which object is shown in the figure? Choose one correct option. Options: {options_text}.",
        "type": "multiple_choice",
        "aspect": "object"
    },
    "T240": {
        "id": "T240",
        "question": "Which USPC class is a patent with the figure from? Choose one correct option. Options: {options_text}.",
        "type": "multiple_choice",
        "aspect": "uspc",
    },
    "T310": {
        "id": "T310",
        "question": "What is the type of the figure shown? Provide the class label.",
        "type": "open_ended",
        "aspect": "type",
    },
    "T320": {
        "id": "T320",
        "question": "What is the projection of the figure shown? Provide the class label.",
        "type": "open_ended",
        "aspect": "projection",
    },
    "T330": {
        "id": "T330",
        "question": "What object is shown in the figure? Provide the class label.",
        "type": "open_ended",
        "aspect": "object"
    },
    "T340": {
        "id": "T340",
        "question": "What USPC class is a patent with the shown figure from? Provide the class label.",
        "type": "open_ended",
        "aspect": "uspc"
    }
}