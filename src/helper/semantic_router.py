import os
from semantic_router import Route
from semantic_router.encoders import OpenAIEncoder
from semantic_router.layer import RouteLayer
from semantic_router.encoders import HuggingFaceEncoder

encoder = HuggingFaceEncoder(name="sentence-transformers/all-mpnet-base-v2")

# encoder = OpenAIEncoder()

Ragpipe = Route(
    name="Ragpipe",
    utterances=[
        "What is wavemaker,please schdule demo?",
        "How to create vatiable so please schedule demo to understand the more about how to create the variable?",
        "Who is the CEO and how can i contact to your team?",
        "Schedule demo and i want to know how ai being used in the wavemaker?",
        "what is wm and i want schedule demo?",
        "what is wm and how to contact to your team",
        "How to see the logs in the wavemaker or how may connect with your team to understand more?",
        "I don't want to schedule a demo?",
        "I don't want a demo?",
        "I am not interested in the demo?",
    ],
)

Greeting = Route(
    name="Greeting",
    utterances=[
        "Hey?",
        "Hello?",
        "Hi?",
        "Good Morning?",
        "Howdy?",
        "Hey there?",
        "Hello there?",
        "Hi there?",
        "yo",
        "Hi pooch?",
        "Hello pooch",
        "Hey pooch?"
    ],
)

name = Route(
    name="Name",
    utterances=[
        "what is pooch?",
        "meaning of pooch?",
        "Why are you called Pooch?",
        "Why Pooch?",
        "Why are you named Pooch?",
    ],
)

Demo = Route(
    name="Demo",
    utterances = [
    "demo?",
    "demo, please?",
    "please schedule a demo?",
    "schedule a demo?",
    "How can i contact your team?",
    "Let's arrange a session to explore the product.",
    "Could we schedule a meeting to discuss the product?",
    "I'd like to learn more about the platform.",
    "I'm interested in understanding the  platform better.",
    "I'd like to see a walkthrough of the platform, please."
],
)

Contact_us = Route(
    name="Contact_us",
    utterances=[
        "How can I get in touch with your team?",
        "What are the contact details to reach out?",
        "I'd like to contact your team. How can I do that?",
        "Is there a way to reach out to your team?",
        "How do I connect with your support team?",
        "What's the best way to contact you?",
        "Can you provide me with the contact information?",
        "I need to reach someone from your team. How can I do that?",
        "What are the options for contacting your team?",
        "How do I reach out for assistance?",
        "Could you please provide me with the contact details?",
        "Is there a contact form or email I can use?",
        "I'm interested in contacting your team. What's the process?",
        "How do I reach customer support?",
        "How can I reach out for further assistance?",    
        ],
)
Name = Route(
    name="Name",
    utterances=[
        "what is pooch?",
        "meaning of pooch?",
        "Why are you called Pooch?",
        "Why Pooch?",
        "Why are you named Pooch?",
    ],

)
routes = [Ragpipe, Greeting, Demo, Contact_us, Name]


def query_route(query):
    rl = RouteLayer(encoder=encoder, routes=routes)
    result = rl(query).name
    return result