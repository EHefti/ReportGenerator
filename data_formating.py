import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.firefox.service import Service as FirefoxService
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions


def df_formatting(df_raw):
    """
    Function to format the DataFrame by converting specific columns to numeric types.
    """
    # --- Data Cleaning Step 1: Delete Unnamed Columns ---
    # Find columns named like 'Unnamed: X' (pandas default for empty headers)

    unnamed_columns = df_raw.columns[df_raw.columns.str.contains('^Unnamed:')]

    if unnamed_columns.any():
        # print(f"\nFound unnamed columns: {unnamed_columns.tolist()}")
        # Drop unnamed columns
        df_cleaned = df_raw.drop(columns=unnamed_columns)
        print(f"/nDropped unnamed columns.")


    # --- Data Cleaning Step 2: Create Second DF with Header and First Data Row ---
    df_names = df_cleaned.copy().iloc[0]
    print(f'/nThe column_id is mapped to the following Information: {df_names}')



    # --- Data Cleaning Step 3: Drop the first 7 rows of the cleaned df ---
    df_cleaned = df_cleaned.iloc[7:]
    print(f'\nThe first 5 rows of the cleaned df: \n{df_cleaned.head()}')
    return df_cleaned, df_names



def clean_data(data):
    """
    Cleans the data by removing unwanted characters and converting to float.

    Args:
        data (pd.Series): The data to clean.

    Returns:
        pd.Series: Cleaned data.
    """
    # Remove unwanted characters and convert to float
    data = data.astype(str).str.replace('%', '', regex=False)
    data = data.astype(str).str.replace(',', '.', regex=False)  # Replace comma with dot
    data = data.str.replace(' ', '', regex=False)  # Remove spaces
    data = data.str.replace('-', '', regex=False)  # Remove dashes
    data = data.str.replace('(', '', regex=False)  # Remove opening parentheses
    data = data.str.replace('0', '', regex=False)  # Remove leading zeros

    return pd.to_numeric(data, errors='coerce')    # Convert to float


def save_styled_dataframe_to_png(styled_df, output_path="styled_table.png", width=None, height=None,
                                row_name_width_px=150, min_column_width_px=100):
    """
    Saves a pandas Styler object to a PNG image using Selenium, ensuring
    full column titles are displayed (potentially wrapping to multiple lines),
    with adjustable maximum window size.

    Args:
        styled_df: The pandas Styler object.
        output_path (str): The path to save the PNG image.
        row_name_width_px (int, optional): The width allocated to the row names (index) in pixels. Defaults to 150.
        min_column_width_px (int, optional): The minimum width of each data column in pixels.
                                             The browser can expand this if needed for title wrapping. Defaults to 100.
        max_width_px (int, optional): The maximum width of the browser window in pixels.
                                      If None, the width will adjust to the content. Defaults to None.
        max_height_px (int, optional): The maximum height of the browser window in pixels.
                                       If None, the height will adjust to the content. Defaults to None.
    """
    html = styled_df.to_html()

    # Inject CSS to control row name width and allow column titles to wrap
    html = f"""
    <style>
      .index_name {{
        min-width: {row_name_width_px}px;
        word-wrap: break-word;
        text-align: left;
        vertical-align: top;
      }}
      .col_heading {{
        min-width: {min_column_width_px}px;
        word-wrap: break-word;
        text-align: center;
        vertical-align: top;
        padding: 2px;
        font-size: 16px;
        font-weight: normal;
        /*transform: rotate(-90deg);*/
        /*transform-origin: top left;*/
      }}
      th {{
        min-width: {row_name_width_px}px;
        word-wrap: break-word;
        vertical-align: top;
        padding: 5px;
        text-align: left;
      }}
      td {{
        min-width: {min_column_width_px}px;
        word-wrap: break-word;
        text-align: center;
        vertical-align: top;
        padding: 5px;
      }}
    </style>
    {html}
    """

    browser_name = "chrome"  # Or "firefox"

    if browser_name == "chrome":
        chrome_options = ChromeOptions()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument(f"--force-device-scale-factor=5")  # Set the scale factor
        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
    elif browser_name == "firefox":
        firefox_options = FirefoxOptions()
        firefox_options.add_argument("--headless")
        firefox_options.set_preference("layout.css.devPixelsPerPx", "2")
        service = FirefoxService(GeckoDriverManager().install())
        driver = webdriver.Firefox(service=service, options=firefox_options)
    else:
        raise ValueError(f"Unsupported browser: {browser_name}")

    try:
        driver.get(f"data:text/html;charset=utf-8,{html}")
        time.sleep(1)

        element = driver.find_element(By.TAG_NAME, "body")
        content_width = element.size['width'] + row_name_width_px + 20 # Add some extra for borders/padding
        content_height = element.size['height'] + 20

        window_width = content_width
        window_height = content_height

        if width is not None and content_width > width:
            window_width = width
        elif width is not None:
            window_width = max(content_width, width) # Ensure at least max_width if set

        if height is not None and content_height > height:
            window_height = height
        elif height is not None:
            window_height = max(content_height, height) # Ensure at least max_height if set

        driver.set_window_size(int(window_width), int(window_height))
        time.sleep(1)
        element = driver.find_element(By.TAG_NAME, "body")
        element.screenshot(output_path)
        print(f"Styled DataFrame saved to: {output_path} with dimensions: {element.size} (Window size: {int(window_width)}x{int(window_height)})")

    finally:
        driver.quit()