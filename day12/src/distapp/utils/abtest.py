#create a/b test for two versoins of the web application
import random
def ab_test():
   #A is older version, B is newer version
   user_a=10000
   conversion_a=500
   user_b=10000
   conversion_b=600
   rate_a=conversion_a/user_a
   rate_b=conversion_b/user_b   
   pooled_rate=(conversion_a+conversion_b)/(user_a+user_b)
   standard_error=(pooled_rate*(1-pooled_rate)*(1/user_a+1/user_b))**0.5
   z_score=(rate_b-rate_a)/standard_error
   p_value=1-0.5*(1+z_score/abs(z_score))
   return z_score, p_value

if __name__ == "__main__":
    z_score, p_value=ab_test()
    print(f"Z-score: {z_score}, P-value: {p_value}")
    if p_value<0.05:
        print("Reject null hypothesis, version B is better than version A") 
    else:
        print("Fail to reject null hypothesis, no significant difference between version A and version B")
