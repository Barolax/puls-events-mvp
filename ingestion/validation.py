import great_expectations as gx
import pandas as pd
from dotenv import load_dotenv

load_dotenv(override=True)


def validate_events(events: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Valide les événements avec Great Expectations.
    Retourne (événements valides, événements invalides).
    """
    df = pd.DataFrame(events)

    context = gx.get_context()

    datasource = context.data_sources.add_pandas("events_source")
    asset = datasource.add_dataframe_asset("events_asset")
    batch_definition = asset.add_batch_definition_whole_dataframe("batch")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    suite = context.suites.add(gx.ExpectationSuite(name="events_suite"))

    # Règle 1 — titre non vide
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="title"))

    # Règle 2 — ville renseignée
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="city"))

    # Règle 3 — coordonnées GPS valides
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="latitude", min_value=-90, max_value=90))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="longitude", min_value=-180, max_value=180))

    validation_definition = context.validation_definitions.add(
        gx.ValidationDefinition(name="events_validation", data=batch_definition, suite=suite)
    )

    results = validation_definition.run(batch_parameters={"dataframe": df})

    # Séparation valides / invalides
    invalid_ids = set()
    for result in results.results:
        if not result.success:
            unexpected_index = result.result.get("unexpected_index_list", [])
            for idx in unexpected_index:
                invalid_ids.add(df.iloc[idx]["id"])

    valid_events = [e for e in events if e["id"] not in invalid_ids]
    invalid_events = [e for e in events if e["id"] in invalid_ids]

    print(f"Validation : {len(valid_events)} valides, {len(invalid_events)} invalides")
    return valid_events, invalid_events


if __name__ == "__main__":
    from open_agenda import fetch_all_events

    events = fetch_all_events(size_per_agenda=10)
    valid, invalid = validate_events(events)

    if invalid:
        print(f"\nÉvénements invalides :")
        for e in invalid[:3]:
            print(f"  - {e['title']} | ville: {e['city']} | lat: {e['latitude']}")