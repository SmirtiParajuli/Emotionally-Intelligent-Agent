import random

def generate_agent_response(text, emotion):
    """
    Generates an emotionally-aligned agent response by combining the original input text
    with a randomly selected predefined reply matching the detected emotion.
    
    Args:
        text (str): The original user input or base message.
        emotion (str): The detected emotion (e.g., 'happy', 'sad', 'angry').

    Returns:
        str: A combined message that includes the original text and an emotionally appropriate reply.
    """
    emotion = emotion.lower()
    base = f"{text} "

    responses = {
        "happy": [
            f"{base}😄 That really makes me smile!",
            f"{base}✨ I'm feeling joyful just thinking about it!",
            f"{base}😃 This is wonderful news!"
        ],
        "sad": [
            f"{base}😢 That’s really hard to say...",
            f"{base}😢 I'm feeling a bit down.",
            f"{base}😭 It's tough to express this, but it matters."
        ],
        "angry": [
            f"{base}😠 I'm upset, but this needs to be said.",
            f"{base}🔥 I'm seriously frustrated!",
            f"{base}😤 This isn’t right and I have to say it."
        ],
        "calm": [
            f"{base}😌 I’m sharing this peacefully.",
            f"{base}🌿 Let’s stay composed while I tell you this.",
            f"{base}🧘 I’ve thought about this carefully and calmly."
        ],
        "surprised": [
            f"{base}😲 Whoa—I didn’t expect this!",
            f"{base}😮 This caught me completely off guard!",
            f"{base}🤯 Wow! That’s a big surprise."
        ],
        "fearful": [
            f"{base}😨 I’m scared to say this, but I must.",
            f"{base}😰 Please don’t panic... I’m really worried.",
            f"{base}😱 This makes me anxious to admit."
        ],
        "disgusted": [
            f"{base}😒 Honestly... this disgusts me.",
            f"{base}🤢 I feel very uncomfortable even talking about it.",
            f"{base}🤮 Sorry, this makes me sick to say."
        ],
        "nervous": [
            f"{base}😬 I’m not sure how you’ll take this...",
            f"{base}🫣 I'm nervous just saying it.",
            f"{base}😓 My hands are shaking a little..."
        ],
        "neutral": [
            f"{base}🤖 Here’s what I want to tell you.",
            f"{base}📢 I'm just sharing the facts.",
            f"{base}🗣️ Just giving you this info plainly."
        ]
    }

    return random.choice(responses.get(emotion, responses["neutral"]))

"""
This function simulates empathetic agent behavior by tailoring replies based on the user's emotional state. The use of tone-aware phrasing and emojis creates 
a more engaging interaction. In future iterations, this could evolve into a dialogue system that adapts personality, tone, and depth over time — enabling agents
with deeper social intelligence.
"""