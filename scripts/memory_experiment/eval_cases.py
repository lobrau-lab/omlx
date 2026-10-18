from typing import List, Dict, Any

EVAL_CASES = [
    {
        "name": "Old detail present",
        "description": "A detail present in one old turn",
        "history": [
            {"role": "user", "content": "I have a dog named Rufus. He is a golden retriever."},
            {"role": "assistant", "content": "That's lovely! Golden retrievers are great dogs."},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "user", "content": "I like eating apples."},
            {"role": "assistant", "content": "Apples are a very healthy snack."},
            {"role": "user", "content": "Tell me a joke."},
            {"role": "assistant", "content": "Why did the scarecrow win an award? Because he was outstanding in his field!"},
            {"role": "user", "content": "What is my dog's name?"}
        ],
        "expected_answer_concepts": ["Rufus"],
        "requires_retrieval": True,
        "should_abstain": False
    },
    {
        "name": "Multiple turns required",
        "description": "A question requiring multiple turns to answer completely",
        "history": [
            {"role": "user", "content": "My favorite color is blue."},
            {"role": "assistant", "content": "Blue is a very calming color."},
            {"role": "user", "content": "Also, my favorite shape is a triangle."},
            {"role": "assistant", "content": "Triangles are strong structural shapes."},
            {"role": "user", "content": "And my favorite number is 7."},
            {"role": "assistant", "content": "7 is considered lucky by many."},
            {"role": "user", "content": "If I draw my favorite shape using my favorite color, and put my favorite number inside it, what would it look like?"}
        ],
        "expected_answer_concepts": ["blue", "triangle", "7"],
        "requires_retrieval": True,
        "should_abstain": False
    },
    {
        "name": "Preference change",
        "description": "A preference that changes over time",
        "history": [
            {"role": "user", "content": "I love drinking coffee in the morning."},
            {"role": "assistant", "content": "Coffee is a great way to wake up."},
            {"role": "user", "content": "Can you recommend a breakfast?"},
            {"role": "assistant", "content": "How about oatmeal and a nice cup of coffee?"},
            {"role": "user", "content": "Actually, I stopped drinking coffee. I only drink tea now."},
            {"role": "assistant", "content": "Oh, good to know! I'll remember you prefer tea."},
            {"role": "user", "content": "What should I have to drink with breakfast?"}
        ],
        "expected_answer_concepts": ["tea"],
        "unexpected_answer_concepts": ["coffee"],
        "requires_retrieval": True,
        "should_abstain": False
    },
    {
        "name": "Explicit correction",
        "description": "An explicit correction of a previous fact",
        "history": [
            {"role": "user", "content": "My brother's name is Mark."},
            {"role": "assistant", "content": "Got it, your brother is Mark."},
            {"role": "user", "content": "Wait, I made a typo. My brother's name is actually Matt."},
            {"role": "assistant", "content": "Ah, okay. Your brother is Matt."},
            {"role": "user", "content": "Can you remind me what my brother's name is?"}
        ],
        "expected_answer_concepts": ["Matt"],
        "unexpected_answer_concepts": ["Mark"],
        "requires_retrieval": True,
        "should_abstain": False
    },
    {
        "name": "Ambiguous contradiction",
        "description": "An ambiguous contradiction where clarification is needed",
        "history": [
            {"role": "user", "content": "I live in New York."},
            {"role": "assistant", "content": "New York is a bustling city!"},
            {"role": "user", "content": "I am looking for a new apartment in Los Angeles."},
            {"role": "assistant", "content": "Los Angeles has some great neighborhoods. What kind of place are you looking for?"},
            {"role": "user", "content": "Where do I currently live?"}
        ],
        "expected_answer_concepts": [],
        "requires_retrieval": True,
        "should_abstain": True # The model should ask for clarification or mention both
    },
    {
        "name": "Absent from history",
        "description": "A question whose answer is absent from history",
        "history": [
            {"role": "user", "content": "I bought a new car yesterday."},
            {"role": "assistant", "content": "Congratulations on the new car!"},
            {"role": "user", "content": "It drives really smoothly."},
            {"role": "assistant", "content": "That's wonderful to hear."},
            {"role": "user", "content": "What color is my new car?"}
        ],
        "expected_answer_concepts": [],
        "requires_retrieval": True,
        "should_abstain": True # Model shouldn't know the color
    },
    {
        "name": "Recent turns alone",
        "description": "A question answerable from recent turns alone",
        "history": [
            {"role": "user", "content": "Let's talk about space."},
            {"role": "assistant", "content": "Space is vast and fascinating. What would you like to know?"},
            {"role": "user", "content": "What is the closest planet to the sun?"},
            {"role": "assistant", "content": "The closest planet to the sun is Mercury."},
            {"role": "user", "content": "Is it the hottest planet?"}
        ],
        "expected_answer_concepts": ["Venus"], # Venus is hotter, context is Mercury
        "requires_retrieval": False,
        "should_abstain": False
    }
]
