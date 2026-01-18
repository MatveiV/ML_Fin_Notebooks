#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тестовый скрипт для автоматической проверки генерации PDF
"""

import sys
from pathlib import Path
from pdf_generator import PDFGenerator

def test_pdf_generation():
    """Тестирование генерации PDF"""
    print("Тестирование PDF генерации...")

    generator = PDFGenerator()

    # Загружаем CSV данные
    csv_file = Path("data/invoices.csv")
    df = generator.load_data_file(csv_file)
    print(f"Загружено {len(df)} записей из CSV")

    # Получаем invoice IDs
    invoice_ids = generator.get_invoice_ids(df)
    print(f"Найдены invoice IDs: {invoice_ids[:3]}...")  # Показываем первые 3

    # Загружаем шаблон
    template_file = Path("templates/invoice_template.html")
    template_html = generator.load_template(template_file)
    print("Шаблон загружен")

    # Генерируем PDF для первого invoice
    first_invoice_id = invoice_ids[0]
    invoice_data = generator.get_invoice_data(df, first_invoice_id)

    output_path = Path("output/test_invoice.pdf")
    generator.generate_pdf(template_html, invoice_data, output_path)

    print(f"PDF сгенерирован: {output_path}")
    print(f"Размер файла: {output_path.stat().st_size} байт")

    # Тестируем с JSON данными
    json_file = Path("data/invoices.json")
    df_json = generator.load_data_file(json_file)
    print(f"Загружено {len(df_json)} записей из JSON")

    json_invoice_ids = generator.get_invoice_ids(df_json)
    first_json_invoice = generator.get_invoice_data(df_json, json_invoice_ids[0])

    output_path_json = Path("output/test_invoice_json.pdf")
    generator.generate_pdf(template_html, first_json_invoice, output_path_json)

    print(f"PDF из JSON сгенерирован: {output_path_json}")
    print(f"Размер файла: {output_path_json.stat().st_size} байт")

    print("\nВсе тесты пройдены успешно!")

if __name__ == "__main__":
    test_pdf_generation()