"""USDA FoodData Central API wrapper.

TODO:
- Add API key configuration.
- Implement request and cache helpers.
- Map model labels to USDA search queries.
"""


class USDALookupClient:
    """Thin wrapper around USDA nutrition lookup."""

    def search(self, food_name: str):
        raise NotImplementedError("Implement USDA food search.")
