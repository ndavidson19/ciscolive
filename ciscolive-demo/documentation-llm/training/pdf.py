#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import os
import re
from pdfminer.layout import LTTextBoxHorizontal, LAParams
from pdfminer.high_level import extract_pages
from collections import Counter
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument

def extract_title_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        parser = PDFParser(f)
        doc = PDFDocument(parser)
        
        metadata = doc.info[0]
        title = metadata.get('Title', None) if metadata else None
        if isinstance(title, bytes):
            title = title.decode('utf-8')
        return title


def extract_text_with_structure(filename, font_size_threshold=12, source_id=None):
    """
    Extracts text from a PDF with a basic structure distinguishing between headers and content.
    :param filename: Path to the PDF file.
    :param font_size_threshold: Font size to be used as a threshold to distinguish headers from content.
    :param source_id: Identifier for the source document.
    :return: A list of dictionaries with 'header', 'content', and 'source_id' keys.
    """
    structured_data = []
    current_header = None
    current_content = []
    unwanted_symbols = ['']  
    for page_layout in extract_pages(filename):
        for element in page_layout:
            if isinstance(element, LTTextBoxHorizontal):
                for text_line in element:
                    # Extract font sizes of all characters in the line
                    font_sizes = [char.size for char in text_line if hasattr(char, 'size')]
                    
                    # Use mode as the line's font size
                    most_common_font_size = Counter(font_sizes).most_common(1)[0][0] if font_sizes else 0

                    # Extract text content
                    text_content = text_line.get_text().strip()

                    # Check for headers using the font size threshold
                    if most_common_font_size > font_size_threshold and text_content not in unwanted_symbols:
                        if current_header:
                            structured_data.append({
                                'header': current_header,
                                'content': ' '.join(current_content),
                                'source_id': source_id
                            })
                            current_content = []
                        current_header = text_content
                    else:
                        current_content.append(text_content)

    # Add the last header and content
    if current_header:
        structured_data.append({
            'header': current_header,
            'content': ' '.join(current_content),
            'source_id': source_id
        })

    return structured_data

def cleanup_individual_document(doc):
    print(doc[1])
    # Remove lines with copyright notices followed by pagination info
    copyright_pagination_pattern = r'^© \d{4} Cisco and/or its affiliates\. All rights reserved\. Page \d+ of \d+\s*'
    doc = re.sub(copyright_pagination_pattern, '', doc, flags=re.MULTILINE)

    # Remove specific header-content pairs
    headers_to_remove = ["Table of Contents", "Legal Information", "Documentation Feedback", "Related Content", "Trademarks", "Contents"]
    for header in headers_to_remove:
        doc = re.sub(rf'^{re.escape(header)}\n.*$', '', doc, flags=re.MULTILINE)

    # Process the lines for header-content pairs
    lines = doc.splitlines()
    cleaned_lines = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # If the current line is a header and has meaningful content on the next line
        if line and (i + 1) < len(lines) and '.' in lines[i + 1]:
            cleaned_lines.append(line)  # Header
            cleaned_lines.append(lines[i + 1].strip())  # Content
            cleaned_lines.append('')  # New line separator
            i += 2
        else:
            i += 1

    return '\n'.join(cleaned_lines)

def main():
    # [Part 1: Text Extraction]
    pdf_files = [file for file in os.listdir('./docs/pdfs') if file.endswith('.pdf')]
    all_data = []

    for i, filename in enumerate(pdf_files):
        full_path = os.path.join('./docs/pdfs', filename)
        print(f"Processing file {i + 1}: {filename}")
        
        # Extract title from the PDF and use as source_id
        title = extract_title_from_pdf(full_path)
        source_id = title if title else "Default Source - Unknown Document Type"
        print(source_id)

        try:
            structured_data = extract_text_with_structure(full_path, source_id=source_id)
            all_data.append(structured_data)
        except Exception as e:
            print(f"Error processing {filename}. Reason: {e}")

    # [Part 2: Text Cleaning]
    cleaned_documents = []

    for doc_data in all_data:
        # Combine header-content pairs into one document
        print("Cleaning parsed text into structured format")
        doc_content = "\n".join([f"Source: {item['source_id']}\n{item['header']}\n{item['content']}" for item in doc_data])
        cleaned_doc = cleanup_individual_document(doc_content.strip())
        if cleaned_doc:
            cleaned_documents.append(cleaned_doc)

    cleaned_content = "\n\n".join(cleaned_documents)
    cleaned_content_per_doc = re.sub(r'\n\n+', '\n\n', cleaned_content)
    print("Document cleaned succesfully!")

    # Writing directly to the final cleaned file
    with open('cleaned_file.txt', 'w') as f:
        f.write(cleaned_content_per_doc)
    
    print(f"Processed {len(cleaned_documents)} documents.")
    print("Script execution completed successfully.")
    print("File: cleaned_text.txt has been created and is ready for use in generating embeddings.")


if __name__ == '__main__':
    main()
