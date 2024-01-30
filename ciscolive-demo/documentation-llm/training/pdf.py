import os
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBoxHorizontal, LTTextLine

def extract_text_with_structure(filename):
    structured_data = []
    current_header = None
    current_content = []

    for page_layout in extract_pages(filename):
        for element in page_layout:
            if isinstance(element, LTTextBoxHorizontal):
                for text_line in element:
                    # Extract font sizes of all characters in the line
                    font_sizes = [char.size for char in text_line if hasattr(char, 'size')]
                    # Take the most common font size in the line as the line's font size
                    if font_sizes:
                        avg_font_size = sum(font_sizes) / len(font_sizes)
                    else:
                        avg_font_size = 0

                    # This is a basic check for headers; adjust as per your document's styling
                    if avg_font_size > 12:
                        if current_header:
                            structured_data.append({
                                'header': current_header,
                                'content': ' '.join(current_content)
                            })
                            current_content = []
                        current_header = text_line.get_text().strip()
                    else:
                        current_content.append(text_line.get_text().strip())

    # Adding the last header and content
    if current_header:
        structured_data.append({
            'header': current_header,
            'content': ' '.join(current_content)
        })

    return structured_data


def main():
    pdf_files = [file for file in os.listdir('./docs/pdfs') if file.endswith('.pdf')]
    all_data = []

    for i, filename in enumerate(pdf_files):
        full_path = os.path.join('./docs/pdfs', filename)
        print(f"Processing file {i + 1}: {filename}")

        structured_data = extract_text_with_structure(full_path)
        all_data.extend(structured_data)

    with open('./docs/cisco_docs.txt', 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(item['header'] + "\n")
            f.write(item['content'] + "\n\n")

    print("Script execution completed successfully.")

if __name__ == '__main__':
    main()
