#create estimation.py for sales_data using sales_amount column
import pandas as pd
from scipy import stats
from distapp.configurations.config import Config
#point estimation for sales_amount and purchase_amount columns
def estimation():
    config = Config()
    data = pd.read_csv(config.estimation_path)
    sales_amount = data['sales_amount']
    
    # Calculate mean and standard deviation
    mean_sales = sales_amount.mean()
    std_dev_sales = sales_amount.std()
    
    # calculate mean and standard deviation for purchase_amount
    purchase_amount = data['purchase_amount']
    mean_purchase = purchase_amount.mean()
    std_dev_purchase = purchase_amount.std()

    return mean_sales, std_dev_sales, mean_purchase, std_dev_purchase
    
if __name__ == "__main__":
    mean_sales, std_dev_sales, mean_purchase, std_dev_purchase = estimation()
    print(f"Sales Amount - Mean: {mean_sales}, Standard Deviation: {std_dev_sales}")
    print(f"Purchase Amount - Mean: {mean_purchase}, Standard Deviation: {std_dev_purchase}")
