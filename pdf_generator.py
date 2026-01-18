#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF Generator - инструмент для генерации PDF документов из CSV/JSON данных с использованием HTML шаблонов
"""

import os
import sys
import json
import platform
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from jinja2 import Template
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration


class PDFGenerator:
    def __init__(self):
        # Подавляем GLib предупреждения для Windows
        import os
        os.environ['GIO_USE_VFS'] = 'local'
        os.environ['GIO_MODULE_DIR'] = ''

        self.data_dir = Path("data")
        self.templates_dir = Path("templates")
        self.output_dir = Path("output")

        # Создаем директории, если они не существуют
        self.data_dir.mkdir(exist_ok=True)
        self.templates_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

        # CSS для поддержки кириллицы и формата A4
        self.css = CSS(string="""
            @page {
                size: A4;
                margin: 2cm;
            }
            @font-face {
                font-family: 'DejaVu Sans';
                src: local('DejaVu Sans');
            }
            body {
                font-family: 'DejaVu Sans', 'Arial', sans-serif;
                font-size: 14px;
                line-height: 1.4;
                margin: 0;
                padding: 20px;
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
                border-bottom: 2px solid #333;
                padding-bottom: 20px;
            }
            .invoice-details {
                margin: 20px 0;
            }
            .invoice-details table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            .invoice-details th, .invoice-details td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            .invoice-details th {
                background-color: #f2f2f2;
                font-weight: bold;
            }
            .total {
                font-weight: bold;
                font-size: 16px;
                margin-top: 20px;
            }
        """)

    def get_data_files(self) -> List[Path]:
        """Получить список всех CSV и JSON файлов в директории data"""
        csv_files = list(self.data_dir.glob("*.csv"))
        json_files = list(self.data_dir.glob("*.json"))
        return csv_files + json_files

    def get_template_files(self) -> List[Path]:
        """Получить список всех HTML шаблонов"""
        return list(self.templates_dir.glob("*.html"))

    def load_data_file(self, file_path: Path) -> pd.DataFrame:
        """Загрузить данные из CSV или JSON файла"""
        if file_path.suffix.lower() == '.csv':
            return pd.read_csv(file_path, encoding='utf-8')
        elif file_path.suffix.lower() == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Если JSON содержит список объектов, конвертируем в DataFrame
            if isinstance(data, list):
                return pd.DataFrame(data)
            # Если JSON содержит словарь с данными
            elif isinstance(data, dict) and 'data' in data:
                return pd.DataFrame(data['data'])
            else:
                # Предполагаем, что это словарь с invoice данными
                return pd.DataFrame([data])
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {file_path.suffix}")

    def load_template(self, template_path: Path) -> str:
        """Загрузить HTML шаблон"""
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()

    def get_invoice_ids(self, df: pd.DataFrame) -> List[str]:
        """Получить список уникальных invoice ID"""
        if 'invoice_id' in df.columns:
            return df['invoice_id'].dropna().unique().tolist()
        elif 'id' in df.columns:
            return df['id'].dropna().unique().tolist()
        else:
            # Если нет явного поля ID, используем индексы
            return [str(i) for i in df.index.tolist()]

    def get_invoice_data(self, df: pd.DataFrame, invoice_id: str) -> Dict[str, Any]:
        """Получить данные для конкретного invoice ID"""
        if 'invoice_id' in df.columns:
            invoice_data = df[df['invoice_id'] == invoice_id]
        elif 'id' in df.columns:
            invoice_data = df[df['id'] == invoice_id]
        else:
            # Используем индекс как ID
            try:
                idx = int(invoice_id)
                invoice_data = df.iloc[[idx]]
            except (ValueError, IndexError):
                raise ValueError(f"Invoice ID '{invoice_id}' не найден")

        if invoice_data.empty:
            raise ValueError(f"Invoice ID '{invoice_id}' не найден")

        # Конвертируем в словарь для шаблона
        return invoice_data.iloc[0].to_dict()

    def process_logo_path(self, logo_filename: str) -> str:
        """Обработать путь к логотипу для использования в HTML"""
        if not logo_filename:
            return ""

        # Проверяем различные расширения
        extensions = ['.png', '.jpg', '.jpeg', '.gif', '.svg']
        logos_dir = self.data_dir.parent / "logos"

        for ext in extensions:
            logo_path = logos_dir / f"{logo_filename}{ext}" if not logo_filename.endswith(tuple(extensions)) else logos_dir / logo_filename
            if logo_path.exists():
                # Возвращаем абсолютный путь для WeasyPrint
                return str(logo_path.absolute())

        return ""

    def get_available_logos(self) -> List[str]:
        """Получить список доступных логотипов"""
        logos_dir = self.data_dir.parent / "logos"
        if not logos_dir.exists():
            return []

        # Поддерживаемые расширения
        extensions = ['.png', '.jpg', '.jpeg', '.gif', '.svg']

        logos = []
        for file_path in logos_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                # Возвращаем имя без расширения
                logos.append(file_path.stem)

        return sorted(list(set(logos)))  # Убираем дубликаты

    def select_logo(self, available_logos: List[str]) -> str:
        """Выбрать логотип из списка доступных"""
        if not available_logos:
            print("В папке logos не найдено файлов логотипов.")
            return ""

        print(f"\nНайдено логотипов: {len(available_logos)}")
        logo_names = [f"{logo} (проверены расширения: png, jpg, jpeg, gif, svg)" for logo in available_logos]

        selected_idx = self.print_menu("Выберите логотип компании:", logo_names)
        return available_logos[selected_idx]

    def generate_pdf(self, template_html: str, data: Dict[str, Any], output_path: Path) -> Path:
        """Генерировать PDF из HTML шаблона и данных"""
        try:
            # Проверяем и создаем директорию output, если необходимо
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Обрабатываем путь к логотипу
            if 'company_logo' in data and data['company_logo']:
                data['company_logo'] = self.process_logo_path(data['company_logo'])

            # Создаем шаблон Jinja2
            template = Template(template_html)

            # Рендерим HTML с данными
            html_content = template.render(**data)

            # Подавляем GLib предупреждения, перенаправляя stderr
            import os
            import sys
            from contextlib import redirect_stderr
            import io
            from datetime import datetime

            # Создаем объект HTML из строки с подавлением предупреждений
            stderr_capture = io.StringIO()
            with redirect_stderr(stderr_capture):
                html_doc = HTML(string=html_content)

            # Пытаемся сгенерировать PDF, если файл заблокирован - создаем уникальное имя
            final_output_path = output_path
            attempt = 0
            max_attempts = 5

            while attempt < max_attempts:
                try:
                    # Генерируем PDF с настройками шрифтов и подавлением предупреждений
                    with redirect_stderr(stderr_capture):
                        font_config = FontConfiguration()
                        html_doc.write_pdf(
                            final_output_path,
                            stylesheets=[self.css],
                            font_config=font_config
                        )

                    print(f"PDF успешно создан: {final_output_path}")
                    return final_output_path

                except PermissionError as pe:
                    # Если файл заблокирован, создаем уникальное имя
                    if attempt == 0:
                        print(f"Файл {output_path} заблокирован. Пытаюсь создать файл с уникальным именем...")

                    timestamp = datetime.now().strftime("%H%M%S")
                    stem = output_path.stem
                    suffix = output_path.suffix
                    final_output_path = output_path.parent / f"{stem}_{timestamp}_{attempt + 1}{suffix}"
                    attempt += 1

                except Exception as e:
                    # Для других ошибок не пытаемся повторять
                    raise Exception(f"Ошибка при генерации PDF: {str(e)}")

            # Если все попытки исчерпаны
            raise Exception(f"Не удалось создать PDF файл после {max_attempts} попыток. Последняя ошибка: файл заблокирован")

        except Exception as e:
            if "Ошибка при генерации PDF:" in str(e):
                raise  # Перебрасываем уже обработанную ошибку
            else:
                raise Exception(f"Ошибка при генерации PDF: {str(e)}")

    def open_pdf(self, pdf_path: Path) -> None:
        """Открыть PDF файл в системной программе просмотра"""
        try:
            if platform.system() == "Windows":
                os.startfile(pdf_path)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", pdf_path])
            else:  # Linux
                subprocess.run(["xdg-open", pdf_path])
        except Exception as e:
            print(f"Не удалось автоматически открыть PDF: {str(e)}")
            print(f"Файл сохранен в: {pdf_path}")

    def print_menu(self, title: str, options: List[str]) -> int:
        """Вывести меню и получить выбор пользователя"""
        print(f"\n{title}")
        print("=" * len(title))

        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")

        while True:
            try:
                choice = input("\nВыберите вариант (или 0 для выхода): ").strip()
                if choice == '0':
                    sys.exit(0)

                choice_num = int(choice)
                if 1 <= choice_num <= len(options):
                    return choice_num - 1  # Возвращаем индекс (0-based)
                else:
                    print(f"Пожалуйста, введите число от 1 до {len(options)}")
            except ValueError:
                print("Пожалуйста, введите корректное число")

    def run(self):
        """Основной цикл программы"""
        print("PDF Generator v1.0")
        print("===================")

        # Получаем список файлов данных
        data_files = self.get_data_files()
        if not data_files:
            print("Ошибка: В директории data не найдено CSV или JSON файлов")
            print("Поместите файлы данных в директорию 'data'")
            return

        # Получаем список шаблонов
        template_files = self.get_template_files()
        if not template_files:
            print("Ошибка: В директории templates не найдено HTML шаблонов")
            print("Поместите HTML шаблоны в директорию 'templates'")
            return

        # Выбираем файл данных
        data_file_names = [f.name for f in data_files]
        print(f"\nНайдено файлов данных: {len(data_files)}")
        selected_data_idx = self.print_menu("Выберите файл данных:", data_file_names)
        selected_data_file = data_files[selected_data_idx]

        # Выбираем шаблон
        template_names = [f.name for f in template_files]
        print(f"\nНайдено шаблонов: {len(template_files)}")
        selected_template_idx = self.print_menu("Выберите HTML шаблон:", template_names)
        selected_template = template_files[selected_template_idx]

        # Загружаем данные и шаблон
        try:
            df = self.load_data_file(selected_data_file)
            template_html = self.load_template(selected_template)
        except Exception as e:
            print(f"Ошибка при загрузке файлов: {str(e)}")
            return

        # Получаем список invoice ID
        invoice_ids = self.get_invoice_ids(df)
        if not invoice_ids:
            print("Ошибка: В файле данных не найдено invoice ID")
            return

        # Выбираем invoice ID
        selected_invoice_idx = self.print_menu(
            f"Выберите invoice ID из файла {selected_data_file.name}:",
            invoice_ids
        )
        selected_invoice_id = invoice_ids[selected_invoice_idx]

        # Получаем данные для выбранного invoice
        try:
            invoice_data = self.get_invoice_data(df, selected_invoice_id)
        except Exception as e:
            print(f"Ошибка при получении данных invoice: {str(e)}")
            return

        # Выбираем логотип компании
        available_logos = self.get_available_logos()
        if available_logos:
            selected_logo = self.select_logo(available_logos)
            invoice_data['company_logo'] = selected_logo
        else:
            print("Логотипы не найдены. PDF будет создан без логотипа.")
            invoice_data['company_logo'] = ""

        # Генерируем PDF
        output_filename = f"invoice_{selected_invoice_id}_{selected_data_file.stem}.pdf"
        output_path = self.output_dir / output_filename

        try:
            final_path = self.generate_pdf(template_html, invoice_data, output_path)
            print(f"\nPDF документ создан успешно!")
            print(f"Файл сохранен: {final_path}")

            # Открываем PDF
            self.open_pdf(final_path)

        except Exception as e:
            print(f"Ошибка при генерации PDF: {str(e)}")
            print("Возможные причины:")
            print("- PDF файл открыт в программе просмотра")
            print("- Антивирус блокирует запись файла")
            print("- Недостаточно прав доступа к директории output")
            print("Попробуйте закрыть PDF файл и повторить попытку.")


def main():
    """Точка входа в программу"""
    try:
        generator = PDFGenerator()
        generator.run()
    except KeyboardInterrupt:
        print("\n\nПрограмма прервана пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"\nКритическая ошибка: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()