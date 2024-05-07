import os
from semantic_router import Route
from semantic_router.encoders import OpenAIEncoder
from semantic_router.layer import RouteLayer
from semantic_router.encoders import HuggingFaceEncoder

encoder = HuggingFaceEncoder(name="sentence-transformers/all-mpnet-base-v2")

# encoder = OpenAIEncoder()

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
    ],
)

Demo = Route(
    name="Demo",
    utterances = [
    "schedule demo",
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
routes = [Greeting, Demo]


def query_route(query):
    rl = RouteLayer(encoder=encoder, routes=routes)
    result = rl(query).name
    return result