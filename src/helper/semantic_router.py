from semantic_router import Route
from semantic_router.encoders import OpenAIEncoder
from semantic_router.layer import RouteLayer

encoder = OpenAIEncoder()

Greet = Route(
    name="Greet",
    utterances=[
        "Hey?",
        "Hello?",
        "Hi?",
        "Good Morning?",
        "Howdy?",
    ],
)
routes = [Greet]


def query_route(query):
    rl = RouteLayer(encoder=encoder, routes=routes)
    result = rl(query).name
    return result