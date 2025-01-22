


# from bdd_processor import process_bdd
# from html_processor import process_html
# from mapping import map_bdd_to_html

# def main():
#     # Process BDD scenarios
#     bdd_scenarios = process_bdd("data/bdd_scenarios.txt")

#     # Process HTML pages
#     html_pages = process_html("data/html_pages")

#     # Perform mapping
#     mappings = map_bdd_to_html(bdd_scenarios, html_pages)

#     # Print results
#     for mapping in mappings:
#         print(f"BDD Scenario: {mapping['scenario']}")
#         for page, element in mapping["matches"]:
#             print(f"  Page: {page}, Element: {element['label']}")
#         print()

# if __name__ == "__main__":
#     main()


# from bdd_processor import process_bdd
# from html_processor import process_html
# from mapping import map_bdd_to_html

# def main():
#     # Process BDD scenarios
#     bdd_scenarios = process_bdd("data/bdd_scenarios.txt")

#     # Process HTML pages
#     html_pages = process_html("data/html_pages")

#     # Perform mapping
#     mappings = map_bdd_to_html(bdd_scenarios, html_pages)

#     # Print results
#     for mapping in mappings:
#         print(f"BDD Scenario: {mapping['scenario']}")
#         for match in mapping["matches"]:
#             print(f"  Page: {match['page']}, Step: {match['step']}, Element: {match['element']['label']}")
#         print()

# if __name__ == "__main__":
#     main()

# from bdd_processor import process_bdd
# from html_processor import process_html
# from mapping import map_bdd_to_html

# def main():
#     # Process BDD scenarios
#     bdd_scenarios = process_bdd("data/bdd_scenarios.txt")

#     # Process HTML pages
#     html_pages = process_html("data/html_pages")

#     # Perform mapping
#     mappings = map_bdd_to_html(bdd_scenarios, html_pages)

#     # Print results
#     for mapping in mappings:
#         print(f"BDD Scenario: {mapping['scenario']}")
#         for match in mapping["matches"]:
#             print(f"  Page: {match['page']}, Step: {match['step']}, Element: {match['element']['attributes']['id']}")
#         print()

# if __name__ == "__main__":
#     main()


from bdd_processor import process_bdd
from code_processor import process_html
from mapping import map_bdd_to_html

def main():
    # Process BDD scenarios
    bdd_scenarios = process_bdd("data/bdd_scenarios.txt")

    # Process HTML pages
    html_pages = process_html("data/html_pages")

    # Perform mapping
    mappings = map_bdd_to_html(bdd_scenarios, html_pages)

    # Print results
    for mapping in mappings:
        print(f"BDD Scenario: {mapping['scenario']}")
        for match in mapping["matches"]:
            print(f"  Page: {match['page']}, Step: {match['step']}, Element ID: {match['element']['attributes']['id']}")
        print()

if __name__ == "__main__":
    main()