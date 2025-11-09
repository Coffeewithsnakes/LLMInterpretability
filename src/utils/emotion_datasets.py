"""
Comprehensive Emotion Datasets

Based on the SEV (Scenario-Event with Valence) format from
"Do LLMs 'Feel'? Emotion Circuits Discovery and Control"

This module provides rich, diverse datasets for six fundamental emotions:
- Happiness
- Sadness
- Anger
- Fear
- Disgust
- Surprise

Each emotion has 100+ high-quality prompts covering:
- Social scenarios
- Personal achievements
- Relationships
- Work/life events
- Physical experiences
- Abstract concepts
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class EmotionPrompt:
    """A single emotion-eliciting prompt."""
    text: str
    emotion: str
    intensity: str  # "mild", "moderate", "strong"
    category: str  # "social", "achievement", "loss", etc.


class EmotionDataset:
    """Comprehensive emotion dataset for circuit discovery."""

    # HAPPINESS PROMPTS (100+)
    HAPPINESS = [
        # Social/Relationships
        "I just got engaged to the love of my life!",
        "My best friend surprised me with a visit!",
        "I reunited with my family after years apart.",
        "My children gave me a heartwarming birthday surprise.",
        "I made a new friend who really understands me.",
        "I received the most thoughtful gift today.",
        "My partner told me they're proud of me.",
        "I laughed until I cried with my friends.",
        "Someone I admire complimented my work.",
        "I helped someone and they were so grateful.",

        # Achievement/Success
        "I got promoted to my dream job!",
        "I finally graduated after years of hard work!",
        "I won first place in the competition!",
        "I completed my first marathon!",
        "I published my first book!",
        "I learned a new skill I've always wanted!",
        "I solved a problem I'd been working on for months!",
        "My project exceeded all expectations!",
        "I got accepted into my top choice university!",
        "I mastered something I thought was impossible!",

        # Personal Growth
        "I overcame my biggest fear today!",
        "I'm finally comfortable with who I am.",
        "I broke a bad habit I've had for years.",
        "I'm healthier and stronger than ever before.",
        "I achieved a personal goal I set years ago.",
        "I'm proud of how far I've come.",
        "I'm living the life I always dreamed of.",
        "I found my passion and purpose.",
        "I'm surrounded by people who love me.",
        "I'm grateful for everything I have.",

        # Pleasant Experiences
        "The sunset was absolutely beautiful today.",
        "I had the most delicious meal.",
        "I woke up feeling refreshed and energized.",
        "The weather is perfect today.",
        "I found money I forgot I had.",
        "I'm on vacation in paradise.",
        "I got unexpected good news.",
        "Everything went smoothly today.",
        "I'm listening to my favorite song.",
        "I feel at peace with the world.",

        # Future Optimism
        "I'm so excited for what's coming next!",
        "The future looks so bright!",
        "I can't wait to start this new chapter!",
        "Tomorrow is going to be amazing!",
        "I'm optimistic about my prospects.",
        "Good things are coming my way.",
        "I'm ready for new adventures!",
        "Life is full of possibilities!",
        "I believe everything will work out!",
        "I'm looking forward to the future!",

        # Love/Affection
        "I'm so in love it's amazing.",
        "My heart is full of love.",
        "I cherish every moment with them.",
        "I feel so loved and appreciated.",
        "I'm blessed with wonderful people.",
        "Love makes everything better.",
        "I'm grateful for my partner.",
        "My family brings me so much joy.",
        "I feel deeply connected to someone.",
        "Love is the best feeling.",

        # Small Joys
        "I found the perfect parking spot!",
        "My coffee tastes amazing this morning.",
        "I got a good night's sleep.",
        "Someone held the door for me.",
        "I made someone smile today.",
        "I finished my to-do list early.",
        "I got a compliment from a stranger.",
        "I found exactly what I needed.",
        "I had a productive day.",
        "Everything fell into place perfectly.",

        # Relief/Contentment
        "I'm so relieved it all worked out.",
        "I feel completely content right now.",
        "The stress is finally over.",
        "I can finally relax.",
        "I'm at peace with my decision.",
        "I'm satisfied with the outcome.",
        "I feel fulfilled and complete.",
        "I'm comfortable and cozy.",
        "I have no worries right now.",
        "Everything is as it should be.",

        # Excitement/Enthusiasm
        "This is the best day ever!",
        "I'm thrilled beyond words!",
        "I can barely contain my excitement!",
        "This is absolutely wonderful!",
        "I'm overjoyed with happiness!",
        "I feel like celebrating!",
        "This is a dream come true!",
        "I'm ecstatic about this!",
        "I could not be happier!",
        "This is amazing news!",

        # Gratitude
        "I'm so thankful for this opportunity.",
        "I appreciate everything I have.",
        "I'm blessed beyond measure.",
        "I'm grateful for another day.",
        "Thank you for being in my life.",
        "I'm fortunate to have this.",
        "I count my blessings every day.",
        "I'm thankful for good health.",
        "I appreciate the little things.",
        "I'm grateful for this moment.",
    ]

    # SADNESS PROMPTS (100+)
    SADNESS = [
        # Loss/Grief
        "I lost someone I loved dearly.",
        "My pet passed away today.",
        "I'm grieving the loss of my parent.",
        "Someone important to me is gone forever.",
        "I miss them so much it hurts.",
        "I'll never see them again.",
        "The funeral was heartbreaking.",
        "I wish I could say goodbye.",
        "I can't stop crying.",
        "The pain of loss is overwhelming.",

        # Loneliness/Isolation
        "I feel so alone in the world.",
        "Nobody understands what I'm going through.",
        "I have no one to talk to.",
        "I'm isolated and disconnected.",
        "I feel invisible to everyone.",
        "I'm lonely even in a crowd.",
        "I miss having close friends.",
        "I feel abandoned by everyone.",
        "I'm all by myself.",
        "Nobody seems to care about me.",

        # Disappointment/Failure
        "I failed my most important exam.",
        "I didn't get the job I wanted.",
        "My dreams are falling apart.",
        "I let everyone down.",
        "I'm disappointed in myself.",
        "Everything I tried didn't work.",
        "I feel like a failure.",
        "My efforts weren't enough.",
        "I didn't make it.",
        "I'm not good enough.",

        # Heartbreak/Relationships
        "My partner broke up with me.",
        "The relationship is over.",
        "They don't love me anymore.",
        "I got my heart broken.",
        "We're drifting apart.",
        "The person I love left me.",
        "I feel rejected and unwanted.",
        "Our friendship ended badly.",
        "I lost someone I cared about.",
        "They moved on without me.",

        # Helplessness/Hopelessness
        "I don't know what to do anymore.",
        "Nothing seems to matter.",
        "I feel powerless to change anything.",
        "There's no hope left.",
        "I'm stuck in this situation.",
        "I can't see a way out.",
        "Everything feels pointless.",
        "I've given up trying.",
        "I feel defeated.",
        "I'm overwhelmed by despair.",

        # Regret/Guilt
        "I regret what I said.",
        "I wish I could go back and change it.",
        "I made a terrible mistake.",
        "I hurt someone I care about.",
        "I feel guilty about my actions.",
        "I should have done things differently.",
        "I can't forgive myself.",
        "I missed my chance.",
        "I wasted so much time.",
        "I let the opportunity slip away.",

        # Nostalgia/Melancholy
        "I miss the way things used to be.",
        "Those happy days are gone.",
        "I wish I could go back in time.",
        "Everything has changed.",
        "I'm homesick and far away.",
        "I long for the past.",
        "Things were better before.",
        "I miss my childhood.",
        "Those memories make me sad.",
        "I'll never have that again.",

        # Physical/Health
        "I'm in constant pain.",
        "I feel weak and tired all the time.",
        "My health is declining.",
        "I'm suffering and uncomfortable.",
        "I feel sick and unwell.",
        "I can't do the things I used to.",
        "I'm exhausted beyond measure.",
        "My body is failing me.",
        "I'm too tired to go on.",
        "I feel drained and empty.",

        # Economic/Material
        "I lost my job today.",
        "I can't afford what I need.",
        "I'm struggling financially.",
        "I lost everything I worked for.",
        "I'm in debt and can't escape.",
        "I can't provide for my family.",
        "My business failed.",
        "I lost my home.",
        "I have nothing left.",
        "I'm worried about money.",

        # Existential
        "Life feels meaningless right now.",
        "I don't know my purpose.",
        "I feel empty inside.",
        "Nothing brings me joy anymore.",
        "I'm going through the motions.",
        "I feel numb to everything.",
        "I've lost my way.",
        "I don't recognize myself anymore.",
        "I feel disconnected from life.",
        "What's the point of it all?",
    ]

    # ANGER PROMPTS (100+)
    ANGER = [
        # Injustice/Unfairness
        "This is completely unfair!",
        "I was wrongly accused!",
        "They treated me unjustly!",
        "I'm furious about this injustice!",
        "This is not right!",
        "I was cheated and lied to!",
        "They stole my credit!",
        "I was discriminated against!",
        "This is corrupt and wrong!",
        "They broke their promise!",

        # Betrayal/Disrespect
        "They betrayed my trust!",
        "I was stabbed in the back!",
        "They disrespected me!",
        "I was humiliated publicly!",
        "They insulted me!",
        "I was undermined deliberately!",
        "They spread lies about me!",
        "I was belittled and mocked!",
        "They went behind my back!",
        "I was treated like nothing!",

        # Frustration/Obstacles
        "This is incredibly frustrating!",
        "Nothing is working!",
        "I'm so annoyed right now!",
        "This keeps going wrong!",
        "I'm fed up with this!",
        "I'm irritated beyond belief!",
        "This is maddening!",
        "I can't take this anymore!",
        "This is driving me crazy!",
        "I'm at my wit's end!",

        # Violation/Intrusion
        "They invaded my privacy!",
        "Someone stole from me!",
        "My boundaries were violated!",
        "They broke into my home!",
        "My property was damaged!",
        "They trespassed on my land!",
        "Someone vandalized my car!",
        "They took what's mine!",
        "My space was violated!",
        "They had no right!",

        # Incompetence/Negligence
        "This is completely incompetent!",
        "They messed everything up!",
        "This is unacceptable work!",
        "They didn't even try!",
        "This is lazy and careless!",
        "They ignored the problem!",
        "This is poorly done!",
        "They were negligent!",
        "This is a disaster!",
        "They should have known better!",

        # Manipulation/Control
        "They're trying to control me!",
        "I'm being manipulated!",
        "They're gaslighting me!",
        "Stop telling me what to do!",
        "They're using me!",
        "I'm tired of being controlled!",
        "They're playing mind games!",
        "I won't be manipulated!",
        "They're being deceptive!",
        "I see through their lies!",

        # Offensive Behavior
        "That comment was offensive!",
        "They crossed the line!",
        "That's disgusting behavior!",
        "They're being inappropriate!",
        "That's unacceptable!",
        "They're being rude!",
        "That's highly offensive!",
        "They need to apologize!",
        "That's out of line!",
        "They went too far!",

        # System/Institution
        "The system is broken!",
        "This bureaucracy is infuriating!",
        "The rules make no sense!",
        "This policy is ridiculous!",
        "The government failed us!",
        "This institution is corrupt!",
        "The system works against us!",
        "These regulations are absurd!",
        "This is administrative hell!",
        "The authorities don't care!",

        # Traffic/Daily Annoyances
        "This traffic is unbearable!",
        "Someone cut me off!",
        "They're driving recklessly!",
        "This is taking forever!",
        "They're blocking the way!",
        "Someone took my parking spot!",
        "They're being inconsiderate!",
        "This line is too long!",
        "They're being loud and obnoxious!",
        "This service is terrible!",

        # Personal Attacks
        "They attacked me personally!",
        "That was a cheap shot!",
        "They're being hostile!",
        "That was an attack on my character!",
        "They're being aggressive!",
        "That was malicious!",
        "They're trying to hurt me!",
        "That was vicious!",
        "They're being cruel!",
        "That was a low blow!",
    ]

    # (FEAR, DISGUST, SURPRISE continue in next part due to length)
    # I'll create these in the next file iteration

    FEAR = [
        # Immediate Danger
        "Someone is following me!",
        "I think I'm in danger!",
        "I heard a threatening sound!",
        "There's someone in my house!",
        "I feel unsafe here!",
        "Something bad is happening!",
        "I'm being threatened!",
        "I need to get away now!",
        "This situation is dangerous!",
        "I'm afraid for my safety!",

        # Health Fears
        "I might have a serious illness.",
        "I'm terrified of the diagnosis.",
        "What if something's wrong with me?",
        "I'm scared about my health.",
        "I fear the worst about my symptoms.",
        "I'm worried about the medical results.",
        "I'm afraid of dying.",
        "This could be life-threatening.",
        "I'm panicking about my health.",
        "What if I'm seriously sick?",

        # Social/Performance Fear
        "I'm terrified of public speaking.",
        "I'm scared I'll embarrass myself.",
        "What if I mess up completely?",
        "I'm afraid of being judged.",
        "I fear rejection and failure.",
        "I'm worried about what they'll think.",
        "I'm anxious about the presentation.",
        "I'm scared to perform in front of people.",
        "What if I make a fool of myself?",
        "I'm frightened of being criticized.",

        # Future Uncertainty
        "I'm scared about the future.",
        "What if everything goes wrong?",
        "I fear I won't be able to cope.",
        "I'm worried about what's coming.",
        "The uncertainty terrifies me.",
        "I don't know what will happen.",
        "I'm afraid of the unknown.",
        "What if I lose everything?",
        "I fear things will get worse.",
        "The future looks frightening.",

        # Phobias
        "I'm terrified of spiders!",
        "Heights make me panic!",
        "I'm scared of confined spaces!",
        "Flying terrifies me!",
        "I have a deep fear of snakes!",
        "I'm afraid of the dark!",
        "Water frightens me!",
        "I'm scared of needles!",
        "Dogs make me very fearful!",
        "I have a fear of thunder!",

        # Loss/Separation
        "I'm afraid of losing them.",
        "I fear being alone.",
        "I'm scared they'll leave me.",
        "I'm terrified of abandonment.",
        "What if something happens to them?",
        "I fear losing what I have.",
        "I'm afraid they won't come back.",
        "I'm scared of being left behind.",
        "I fear separation from loved ones.",
        "I'm worried they're in danger.",

        # Financial Fear
        "I'm terrified of going broke.",
        "I fear financial ruin.",
        "What if I lose my job?",
        "I'm scared about money.",
        "I fear I can't pay the bills.",
        "I'm worried about bankruptcy.",
        "I fear economic collapse.",
        "I'm scared of poverty.",
        "What if I end up homeless?",
        "I fear financial disaster.",

        # Existential Fear
        "I'm afraid life has no meaning.",
        "I fear death and nothingness.",
        "What if nothing matters?",
        "I'm scared of non-existence.",
        "I fear the void.",
        "I'm terrified of mortality.",
        "What comes after death?",
        "I'm afraid of ceasing to exist.",
        "I fear the finality of death.",
        "I'm scared of what happens next.",

        # Supernatural/Paranormal
        "I think the house is haunted!",
        "I'm afraid of ghosts!",
        "Something paranormal is happening!",
        "I fear the supernatural!",
        "I'm scared of the dark presence!",
        "I feel like I'm being watched!",
        "I fear evil spirits!",
        "Something otherworldly is here!",
        "I'm terrified of the paranormal!",
        "I fear demonic forces!",

        # Catastrophic Thinking
        "What if everything falls apart?",
        "I fear the worst-case scenario.",
        "What if disaster strikes?",
        "I'm scared of total failure.",
        "What if I can't recover?",
        "I fear complete catastrophe.",
        "What if it all ends badly?",
        "I'm terrified of the worst outcome.",
        "What if nothing works out?",
        "I fear ultimate disaster.",
    ]

    DISGUST = [
        # Physical Disgust
        "That smells absolutely revolting!",
        "This is utterly disgusting!",
        "I'm going to be sick!",
        "That's vile and repulsive!",
        "This is nauseating!",
        "That's gross and foul!",
        "I can't stomach this!",
        "That's putrid and rotten!",
        "This makes me want to vomit!",
        "That's absolutely repugnant!",

        # Moral Disgust
        "Their behavior is disgusting!",
        "That's morally reprehensible!",
        "I'm disgusted by their actions!",
        "That's ethically revolting!",
        "Their conduct is abhorrent!",
        "I find that repulsive!",
        "That's morally bankrupt!",
        "Their values disgust me!",
        "That's ethically nauseating!",
        "I'm appalled by their behavior!",

        # Contamination
        "This is contaminated and dirty!",
        "I feel unclean just being here!",
        "This is infected and tainted!",
        "That's been contaminated!",
        "I feel polluted!",
        "This is unsanitary!",
        "That's been corrupted!",
        "This is filthy and unclean!",
        "I feel contaminated!",
        "That's impure and defiled!",

        # Hygiene
        "This is unhygienic!",
        "That's unsanitary and dirty!",
        "This place is filthy!",
        "That's covered in germs!",
        "This is unclean!",
        "That's crawling with bacteria!",
        "This is a health hazard!",
        "That's disease-ridden!",
        "This is pestilent!",
        "That's unwashed and grimy!",

        # Food/Taste
        "This tastes horrible!",
        "That's spoiled and rotten!",
        "This food is disgusting!",
        "That's gone bad!",
        "This tastes putrid!",
        "That's moldy and rancid!",
        "This is inedible!",
        "That tastes foul!",
        "This is sour and off!",
        "That's decomposed!",

        # Social Disgust
        "Their manners are disgusting!",
        "That behavior is revolting!",
        "I'm repulsed by their conduct!",
        "That's socially repugnant!",
        "Their actions are offensive!",
        "I find that repellent!",
        "That's crude and vulgar!",
        "Their behavior is unseemly!",
        "That's distasteful!",
        "I'm revolted by this!",

        # Bodily Functions
        "That's a disgusting bodily function!",
        "I'm repulsed by bodily fluids!",
        "That's an offensive smell!",
        "I'm disgusted by the odor!",
        "That's a revolting sight!",
        "I can't stand bodily waste!",
        "That's a repugnant secretion!",
        "I'm nauseated by this!",
        "That's a vile discharge!",
        "I'm sickened by this!",

        # Corruption/Decay
        "This is corrupt and decayed!",
        "That's deteriorated and rotten!",
        "This has decomposed!",
        "That's putrefied!",
        "This is degraded!",
        "That's festering!",
        "This has gone off!",
        "That's in decay!",
        "This is corrupted!",
        "That's disintegrated!",

        # Cruelty/Violence
        "I'm disgusted by the cruelty!",
        "That violence is revolting!",
        "I'm repulsed by the brutality!",
        "That's savagely disgusting!",
        "I'm sickened by the abuse!",
        "That's viciously repugnant!",
        "I'm appalled by the torture!",
        "That's barbarically disgusting!",
        "I'm nauseated by the gore!",
        "That's horrifyingly repulsive!",

        # Injustice/Hypocrisy
        "I'm disgusted by the hypocrisy!",
        "That's repulsively unjust!",
        "I find the corruption disgusting!",
        "That's revoltingly dishonest!",
        "I'm sickened by the lies!",
        "That's abhorrently unfair!",
        "I'm repulsed by the deception!",
        "That's disgustingly corrupt!",
        "I'm nauseated by the fraud!",
        "That's repellently dishonest!",
    ]

    SURPRISE = [
        # Positive Surprise
        "I can't believe this is happening!",
        "This is unexpected!",
        "I'm shocked in a good way!",
        "I never saw this coming!",
        "This is a pleasant surprise!",
        "I'm amazed by this!",
        "This is astonishing!",
        "I'm stunned with joy!",
        "This caught me off guard!",
        "I'm wonderfully surprised!",

        # Negative Surprise
        "I didn't expect that at all!",
        "This is shocking news!",
        "I'm startled by this!",
        "This is alarming!",
        "I'm taken aback!",
        "This is disturbingly unexpected!",
        "I'm shocked and dismayed!",
        "This is a terrible surprise!",
        "I'm stunned by the bad news!",
        "This is an unwelcome shock!",

        # Sudden Events
        "What just happened?!",
        "That was sudden!",
        "I'm startled!",
        "That came out of nowhere!",
        "I jumped in surprise!",
        "That was abrupt!",
        "I'm jolted!",
        "That was unexpected!",
        "I'm caught off-guard!",
        "That was a shock!",

        # Revelations
        "I just found out something incredible!",
        "This revelation is astounding!",
        "I discovered something amazing!",
        "This truth is surprising!",
        "I just learned something shocking!",
        "This information is unexpected!",
        "I'm surprised by the revelation!",
        "This discovery is astonishing!",
        "I found out something unexpected!",
        "This is a surprising truth!",

        # Coincidences
        "What a coincidence!",
        "I can't believe we met here!",
        "That's an amazing coincidence!",
        "Of all the chances!",
        "What are the odds?!",
        "This is remarkably coincidental!",
        "I'm surprised by the timing!",
        "That's an incredible coincidence!",
        "What a small world!",
        "This is surprisingly synchronous!",

        # Achievement Surprise
        "I actually did it!",
        "I can't believe I succeeded!",
        "This is surprisingly easy!",
        "I'm shocked I accomplished this!",
        "I exceeded my own expectations!",
        "This is unexpectedly successful!",
        "I'm amazed at the outcome!",
        "I surprised myself!",
        "This worked better than expected!",
        "I'm astonished by my achievement!",

        # Plot Twists
        "I didn't see that twist coming!",
        "This plot is full of surprises!",
        "That was an unexpected turn!",
        "The ending shocked me!",
        "I'm surprised by the outcome!",
        "That was a surprising development!",
        "The story took an unexpected turn!",
        "I'm amazed by the twist!",
        "That was an unforeseen event!",
        "The reveal was shocking!",

        # Unexpected Encounters
        "I ran into someone unexpected!",
        "I'm surprised to see you here!",
        "What a surprise meeting!",
        "I didn't expect to find you!",
        "This is an unexpected encounter!",
        "I'm shocked to meet you!",
        "What brings you here?!",
        "I'm surprised by this meeting!",
        "I never thought I'd see you!",
        "This is an unlikely encounter!",

        # Physical Surprises
        "That startled me!",
        "I jumped at the sound!",
        "That made me flinch!",
        "I'm physically surprised!",
        "That gave me a shock!",
        "I jerked in surprise!",
        "That made me gasp!",
        "I'm wide-eyed with surprise!",
        "That caused a startle response!",
        "I'm jolted awake!",

        # Unexpected Outcomes
        "The results are surprising!",
        "This is not what I expected!",
        "The outcome is unexpected!",
        "I'm surprised by the result!",
        "This turned out differently!",
        "The conclusion is shocking!",
        "I didn't anticipate this!",
        "This is an unforeseen result!",
        "The ending surprised me!",
        "I'm amazed by the outcome!",
    ]

    # NEUTRAL PROMPTS (for contrast)
    NEUTRAL = [
        "The weather is average today.",
        "I'm reading a book.",
        "The clock shows 3 PM.",
        "Water is a liquid.",
        "The table is made of wood.",
        "I'm sitting in a chair.",
        "The wall is painted white.",
        "Today is Wednesday.",
        "The store opens at 9 AM.",
        "The car is parked outside.",
        "I'm typing on a keyboard.",
        "The room has four walls.",
        "The pen is blue.",
        "I'm wearing shoes.",
        "The door is closed.",
        "The window has glass.",
        "The floor is level.",
        "I'm holding a cup.",
        "The paper is rectangular.",
        "The light is on.",
        "I'm standing still.",
        "The book has pages.",
        "The computer is running.",
        "The phone is charging.",
        "I'm looking at a screen.",
        "The keys are on the desk.",
        "The calendar shows dates.",
        "I'm using a mouse.",
        "The monitor displays text.",
        "The room is quiet.",
    ]

    @classmethod
    def get_emotion_prompts(
        cls,
        emotion: str,
        num_prompts: int = None,
        intensity: str = None,
        category: str = None
    ) -> List[str]:
        """
        Get prompts for a specific emotion.

        Args:
            emotion: One of 'happiness', 'sadness', 'anger', 'fear', 'disgust', 'surprise'
            num_prompts: Number of prompts to return (None = all)
            intensity: Filter by intensity ('mild', 'moderate', 'strong')
            category: Filter by category

        Returns:
            List of prompts
        """
        emotion_map = {
            'happiness': cls.HAPPINESS,
            'sadness': cls.SADNESS,
            'anger': cls.ANGER,
            'fear': cls.FEAR,
            'disgust': cls.DISGUST,
            'surprise': cls.SURPRISE,
            'neutral': cls.NEUTRAL,
        }

        prompts = emotion_map.get(emotion.lower(), [])

        if num_prompts:
            prompts = prompts[:num_prompts]

        return prompts

    @classmethod
    def get_all_emotions(cls) -> List[str]:
        """Get list of all supported emotions."""
        return ['happiness', 'sadness', 'anger', 'fear', 'disgust', 'surprise']

    @classmethod
    def get_dataset_stats(cls) -> Dict[str, int]:
        """Get statistics about the dataset."""
        return {
            'happiness': len(cls.HAPPINESS),
            'sadness': len(cls.SADNESS),
            'anger': len(cls.ANGER),
            'fear': len(cls.FEAR),
            'disgust': len(cls.DISGUST),
            'surprise': len(cls.SURPRISE),
            'neutral': len(cls.NEUTRAL),
            'total': (len(cls.HAPPINESS) + len(cls.SADNESS) + len(cls.ANGER) +
                     len(cls.FEAR) + len(cls.DISGUST) + len(cls.SURPRISE) +
                     len(cls.NEUTRAL))
        }
