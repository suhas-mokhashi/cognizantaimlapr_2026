#create customer respository implementation

from kafkamessageapp.models.customer import Customer
from kafkamessageapp.models.full_name import FullName
from kafkamessageapp.repositories.customer_repository import CustomerRepository
from kafkamessageapp.configurations.mysql_conf import session_local

from datetime import datetime
class CustomerRepositoryImpl(CustomerRepository):     
        
     def get_all_customers(self):
          session = session_local()
          try:
                customers = session.query(Customer).all()
                return customers
          except Exception as e:
                raise e
          finally:
                session.close()
    
