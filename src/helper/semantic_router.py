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
        " what is wm and i want schedule demo?",
        "what is wm and how to contact to your team",
        "How to see the logs in the wavemaker or how may connect with your team to understand more?",
        "what is pooch?"
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
        "Hey pooch",
        "Hi pooch",
        "Hello pooch",
        "who are you?"
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
    "I'd like to learn more about the product.",
    "Can we set up a session to go over the product details?",
    "I'm interested in understanding the product better.",
    "When can we arrange a session to explore the product?",
    "Let's plan a meeting to discuss the product further.",
    "I'm eager to learn more. Can we schedule a session?",
    "I'm curious about the product. How about a discussion?",
    "I'm ready to explore the product. Let's have a session.",
    "I'd like to see a walkthrough of the product, please.",
    "Could we arrange a session to go through the product features?",
    "When is a good time to discuss the product?",
    "I'd love to have a conversation about the product.",
    "Let's have a discussion about the product.",
    "I'm interested in exploring the product further.",
    "Could we have a meeting to explore the product together?",
    "Let's schedule a meeting to dive into the product.",
    "I'd like to discuss the product in more detail.",
    "When can we arrange a meeting to talk about the product?",
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
        "How can I reach out for further assistance?",    ],
)
routes = [Ragpipe, Greeting, Demo, Contact_us]


def query_route(query):
    rl = RouteLayer(encoder=encoder, routes=routes)
    result = rl(query).name
    return result