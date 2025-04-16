loan_types = {
    "Home Loan": {
        "interest_rates": {
            "up_to_5_years": 8.99,
            "5_to_10_years": 9.49,
            "above_10_years": 10.00
        }
    },
    "Mortgage Loan": {
        "interest_rates": {
            "up_to_5_years": 10.75,
            "5_to_10_years": 11.25,
            "above_10_years": 11.25
        }
    },
    "Professional Loan": {
        "interest_rates": {
            "up_to_5_years": 10.75
        }
    },
    "Education Loan": {
        "interest_rates": {
            "up_to_5_years": 10.25,
            "5_to_10_years": 10.75,
            "above_10_years": 10.75
        }
    },
    "Hire Purchase Loan": {
        "interest_rates": {
            "up_to_5_years": 10.75,
            "5_to_10_years": 11.25
        }
    },
    "Auto Loan": {
        "interest_rates": {
            "up_to_5_years": 10.25,
            "5_to_10_years": 10.75
        }
    },
    "Siddhartha Hamro Ghar Karja": {
        "interest_rates": {
            "fixed_7_years": 8.75,
            "fixed_7_years_women": 8.49
        }
    },
    "Electric Vehicle": {
        "interest_rates": {
            "default": 9.50,
            "women": 9.25
        }
    }
}

def calculate_emi(principal, annual_rate, tenure_years):
    monthly_rate = annual_rate / (12 * 100)
    tenure_months = tenure_years * 12
    emi = principal * monthly_rate * (1 + monthly_rate)**tenure_months / ((1 + monthly_rate)**tenure_months - 1)
    total_payment = emi * tenure_months
    total_interest = total_payment - principal
    return {
        "monthly_emi": round(emi, 2),
        "total_payment": round(total_payment, 2),
        "total_interest": round(total_interest, 2),
        "yearly_emi": round(emi * 12, 2)
    }
