import pandas as pd

# Store dataset with Westside and Zudio stores
stores_data = [
    # Original Westside Stores
    {
        "Store Name": "Westside Phoenix Marketcity",
        "Location": "Phoenix Marketcity",
        "City": "Mumbai",
        "Store Manager Name": "Priya Sharma",
        "Store Contact Number": "+91-9876543201",
        "Type": "westside"
    },
    {
        "Store Name": "Westside Select City Walk",
        "Location": "Select City Walk Mall",
        "City": "Delhi",
        "Store Manager Name": "Rajesh Kumar",
        "Store Contact Number": "+91-9876543202",
        "Type": "westside"
    },
    {
        "Store Name": "Westside Forum Mall",
        "Location": "Forum Mall",
        "City": "Bangalore",
        "Store Manager Name": "Anita Reddy",
        "Store Contact Number": "+91-9876543203",
        "Type": "westside"
    },
    {
        "Store Name": "Westside Inorbit Mall",
        "Location": "Inorbit Mall",
        "City": "Hyderabad",
        "Store Manager Name": "Suresh Patel",
        "Store Contact Number": "+91-9876543204",
        "Type": "westside"
    },
    {
        "Store Name": "Westside Express Avenue",
        "Location": "Express Avenue Mall",
        "City": "Chennai",
        "Store Manager Name": "Deepa Iyer",
        "Store Contact Number": "+91-9876543205",
        "Type": "westside"
    },
    {
        "Store Name": "Westside Koregaon Park",
        "Location": "Koregaon Park",
        "City": "Pune",
        "Store Manager Name": "Vikram Singh",
        "Store Contact Number": "+91-9876543206",
        "Type": "westside"
    },
    {
        "Store Name": "Westside Salt Lake",
        "Location": "Salt Lake City",
        "City": "Kolkata",
        "Store Manager Name": "Rohini Das",
        "Store Contact Number": "+91-9876543207",
        "Type": "westside"
    },
    {
        "Store Name": "Westside Sector 17",
        "Location": "Sector 17",
        "City": "Chandigarh",
        "Store Manager Name": "Amit Verma",
        "Store Contact Number": "+91-9876543208",
        "Type": "westside"
    },
    
    # NEW WESTSIDE STORES - MUMBAI (5 additional stores)
    {
        "Store Name": "Westside Palladium Mall",
        "Location": "Palladium Mall, Lower Parel",
        "City": "Mumbai",
        "Store Manager Name": "Neha Jain",
        "Store Contact Number": "+91-9876543209",
        "Type": "westside"
    },
    {
        "Store Name": "Westside R City Mall",
        "Location": "R City Mall, Ghatkopar",
        "City": "Mumbai",
        "Store Manager Name": "Karan Malhotra",
        "Store Contact Number": "+91-9876543210",
        "Type": "westside"
    },
    {
        "Store Name": "Westside Infinity Mall",
        "Location": "Infinity Mall, Andheri",
        "City": "Mumbai",
        "Store Manager Name": "Sonal Kapoor",
        "Store Contact Number": "+91-9876543211",
        "Type": "westside"
    },
    {
        "Store Name": "Westside Atria Mall",
        "Location": "Atria Mall, Worli",
        "City": "Mumbai",
        "Store Manager Name": "Rahul Mishra",
        "Store Contact Number": "+91-9876543212",
        "Type": "westside"
    },
    {
        "Store Name": "Westside Seawoods Grand Central",
        "Location": "Seawoods Grand Central, Navi Mumbai",
        "City": "Mumbai",
        "Store Manager Name": "Pooja Agarwal",
        "Store Contact Number": "+91-9876543213",
        "Type": "westside"
    },

    # NEW WESTSIDE STORES - BENGALURU (5 additional stores)
    {
        "Store Name": "Westside UB City Mall",
        "Location": "UB City Mall",
        "City": "Bengaluru",
        "Store Manager Name": "Manoj Rao",
        "Store Contact Number": "+91-9876543214",
        "Type": "westside"
    },
    {
        "Store Name": "Westside Orion Mall",
        "Location": "Orion Mall, Brigade Gateway",
        "City": "Bengaluru",
        "Store Manager Name": "Divya Krishnan",
        "Store Contact Number": "+91-9876543215",
        "Type": "westside"
    },
    {
        "Store Name": "Westside VR Bengaluru",
        "Location": "VR Bengaluru, Whitefield",
        "City": "Bengaluru",
        "Store Manager Name": "Siddharth Hegde",
        "Store Contact Number": "+91-9876543216",
        "Type": "westside"
    },
    {
        "Store Name": "Westside Phoenix Mall of Asia",
        "Location": "Phoenix Mall of Asia",
        "City": "Bengaluru",
        "Store Manager Name": "Prathima Shetty",
        "Store Contact Number": "+91-9876543217",
        "Type": "westside"
    },
    {
        "Store Name": "Westside Mantri Square",
        "Location": "Mantri Square Mall",
        "City": "Bengaluru",
        "Store Manager Name": "Varun Gowda",
        "Store Contact Number": "+91-9876543218",
        "Type": "westside"
    },

    # NEW WESTSIDE STORES - CHENNAI (5 additional stores)
    {
        "Store Name": "Westside Phoenix MarketCity",
        "Location": "Phoenix MarketCity, Velachery",
        "City": "Chennai",
        "Store Manager Name": "Radhika Subramanian",
        "Store Contact Number": "+91-9876543219",
        "Type": "westside"
    },
    {
        "Store Name": "Westside Forum Vijaya Mall",
        "Location": "Forum Vijaya Mall, Vadapalani",
        "City": "Chennai",
        "Store Manager Name": "Arun Kumar",
        "Store Contact Number": "+91-9876543220",
        "Type": "westside"
    },
    {
        "Store Name": "Westside VR Chennai",
        "Location": "VR Chennai, Anna Nagar",
        "City": "Chennai",
        "Store Manager Name": "Meera Natarajan",
        "Store Contact Number": "+91-9876543221",
        "Type": "westside"
    },
    {
        "Store Name": "Westside Ampa Mall",
        "Location": "Ampa Mall, Nelson Manickam Road",
        "City": "Chennai",
        "Store Manager Name": "Ganesh Pillai",
        "Store Contact Number": "+91-9876543222",
        "Type": "westside"
    },
    {
        "Store Name": "Westside Nexus Mall",
        "Location": "Nexus Mall, Korattur",
        "City": "Chennai",
        "Store Manager Name": "Swathi Venkatesh",
        "Store Contact Number": "+91-9876543223",
        "Type": "westside"
    },

    # NEW WESTSIDE STORES - PUNE (5 additional stores)
    {
        "Store Name": "Westside Phoenix Mall",
        "Location": "Phoenix Mall, Viman Nagar",
        "City": "Pune",
        "Store Manager Name": "Ajay Patil",
        "Store Contact Number": "+91-9876543224",
        "Type": "westside"
    },
    {
        "Store Name": "Westside Amanora Mall",
        "Location": "Amanora Mall, Hadapsar",
        "City": "Pune",
        "Store Manager Name": "Sneha Kulkarni",
        "Store Contact Number": "+91-9876543225",
        "Type": "westside"
    },
    {
        "Store Name": "Westside Seasons Mall",
        "Location": "Seasons Mall, Magarpatta",
        "City": "Pune",
        "Store Manager Name": "Rohit Joshi",
        "Store Contact Number": "+91-9876543226",
        "Type": "westside"
    },
    {
        "Store Name": "Westside Elpro City Square",
        "Location": "Elpro City Square, Chinchwad",
        "City": "Pune",
        "Store Manager Name": "Manisha Deshpande",
        "Store Contact Number": "+91-9876543227",
        "Type": "westside"
    },
    {
        "Store Name": "Westside Pavillion Mall",
        "Location": "Pavillion Mall, Shivaji Nagar",
        "City": "Pune",
        "Store Manager Name": "Sachin Bhosale",
        "Store Contact Number": "+91-9876543228",
        "Type": "westside"
    },
    
    # Original Zudio Stores
    {
        "Store Name": "Zudio Koramangala",
        "Location": "Koramangala",
        "City": "Bangalore",
        "Store Manager Name": "Kavya Nair",
        "Store Contact Number": "+91-9876543301",
        "Type": "zudio"
    },
    {
        "Store Name": "Zudio Phoenix Mumbai",
        "Location": "Phoenix Marketcity",
        "City": "Mumbai",
        "Store Manager Name": "Arjun Mehta",
        "Store Contact Number": "+91-9876543302",
        "Type": "zudio"
    },
    {
        "Store Name": "Zudio Jammu",
        "Location": "Jammu City Center",
        "City": "Jammu",
        "Store Manager Name": "Sunita Gupta",
        "Store Contact Number": "+91-9876543303",
        "Type": "zudio"
    },
    {
        "Store Name": "Zudio Bhopal",
        "Location": "DB City Mall",
        "City": "Bhopal",
        "Store Manager Name": "Ravi Joshi",
        "Store Contact Number": "+91-9876543304",
        "Type": "zudio"
    },
    {
        "Store Name": "Zudio Velachery",
        "Location": "Velachery",
        "City": "Chennai",
        "Store Manager Name": "Lakshmi Priya",
        "Store Contact Number": "+91-9876543305",
        "Type": "zudio"
    },
    {
        "Store Name": "Zudio Delhi",
        "Location": "Connaught Place",
        "City": "Delhi",
        "Store Manager Name": "Mohit Agarwal",
        "Store Contact Number": "+91-9876543306",
        "Type": "zudio"
    },
    {
        "Store Name": "Zudio Pune",
        "Location": "Koregaon Park",
        "City": "Pune",
        "Store Manager Name": "Shruti Desai",
        "Store Contact Number": "+91-9876543307",
        "Type": "zudio"
    },
    {
        "Store Name": "Zudio Hyderabad",
        "Location": "Banjara Hills",
        "City": "Hyderabad",
        "Store Manager Name": "Naveen Reddy",
        "Store Contact Number": "+91-9876543308",
        "Type": "zudio"
    },
    {
        "Store Name": "Zudio Chennai",
        "Location": "T. Nagar",
        "City": "Chennai",
        "Store Manager Name": "Preethi Raman",
        "Store Contact Number": "+91-9876543309",
        "Type": "zudio"
    },

    # NEW ZUDIO STORES - MUMBAI (5 additional stores)
    {
        "Store Name": "Zudio Palladium Lower Parel",
        "Location": "Palladium Mall, Lower Parel",
        "City": "Mumbai",
        "Store Manager Name": "Ritesh Shah",
        "Store Contact Number": "+91-9876543310",
        "Type": "zudio"
    },
    {
        "Store Name": "Zudio R City Ghatkopar",
        "Location": "R City Mall, Ghatkopar",
        "City": "Mumbai",
        "Store Manager Name": "Priya Tiwari",
        "Store Contact Number": "+91-9876543311",
        "Type": "zudio"
    },
    {
        "Store Name": "Zudio Infinity Andheri",
        "Location": "Infinity Mall, Andheri",
        "City": "Mumbai",
        "Store Manager Name": "Kunal Sinha",
        "Store Contact Number": "+91-9876543312",
        "Type": "zudio"
    },
    {
        "Store Name": "Zudio Thane",
        "Location": "Viviana Mall, Thane",
        "City": "Mumbai",
        "Store Manager Name": "Ashwini Pawar",
        "Store Contact Number": "+91-9876543313",
        "Type": "zudio"
    },
    {
        "Store Name": "Zudio Navi Mumbai",
        "Location": "Seawoods Grand Central",
        "City": "Mumbai",
        "Store Manager Name": "Ravi Menon",
        "Store Contact Number": "+91-9876543314",
        "Type": "zudio"
    },

    # NEW ZUDIO STORES - BENGALURU (5 additional stores)
    {
        "Store Name": "Zudio UB City",
        "Location": "UB City Mall",
        "City": "Bengaluru",
        "Store Manager Name": "Asha Murthy",
        "Store Contact Number": "+91-9876543315",
        "Type": "zudio"
    },
    {
        "Store Name": "Zudio Orion Mall",
        "Location": "Orion Mall, Brigade Gateway",
        "City": "Bengaluru",
        "Store Manager Name": "Vikram Reddy",
        "Store Contact Number": "+91-9876543316",
        "Type": "zudio"
    },
    {
        "Store Name": "Zudio VR Whitefield",
        "Location": "VR Bengaluru, Whitefield",
        "City": "Bengaluru",
        "Store Manager Name": "Shilpa Rao",
        "Store Contact Number": "+91-9876543317",
        "Type": "zudio"
    },
    {
        "Store Name": "Zudio Phoenix Mall of Asia",
        "Location": "Phoenix Mall of Asia",
        "City": "Bengaluru",
        "Store Manager Name": "Deepak Kumar",
        "Store Contact Number": "+91-9876543318",
        "Type": "zudio"
    },
    {
        "Store Name": "Zudio Mantri Square",
        "Location": "Mantri Square Mall",
        "City": "Bengaluru",
        "Store Manager Name": "Swathi Nair",
        "Store Contact Number": "+91-9876543319",
        "Type": "zudio"
    },

    # NEW ZUDIO STORES - CHENNAI (5 additional stores)
    {
        "Store Name": "Zudio Phoenix MarketCity",
        "Location": "Phoenix MarketCity, Velachery",
        "City": "Chennai",
        "Store Manager Name": "Karthik Raman",
        "Store Contact Number": "+91-9876543320",
        "Type": "zudio"
    },
    {
        "Store Name": "Zudio Forum Vijaya",
        "Location": "Forum Vijaya Mall, Vadapalani",
        "City": "Chennai",
        "Store Manager Name": "Sowmya Krishnan",
        "Store Contact Number": "+91-9876543321",
        "Type": "zudio"
    },
    {
        "Store Name": "Zudio VR Anna Nagar",
        "Location": "VR Chennai, Anna Nagar",
        "City": "Chennai",
        "Store Manager Name": "Ravi Sundar",
        "Store Contact Number": "+91-9876543322",
        "Type": "zudio"
    },
    {
        "Store Name": "Zudio Express Avenue",
        "Location": "Express Avenue Mall",
        "City": "Chennai",
        "Store Manager Name": "Nisha Patel",
        "Store Contact Number": "+91-9876543323",
        "Type": "zudio"
    },
    {
        "Store Name": "Zudio Nexus Korattur",
        "Location": "Nexus Mall, Korattur",
        "City": "Chennai",
        "Store Manager Name": "Arjun Iyer",
        "Store Contact Number": "+91-9876543324",
        "Type": "zudio"
    },

    # NEW ZUDIO STORES - PUNE (5 additional stores)
    {
        "Store Name": "Zudio Phoenix Viman Nagar",
        "Location": "Phoenix Mall, Viman Nagar",
        "City": "Pune",
        "Store Manager Name": "Nikhil Sharma",
        "Store Contact Number": "+91-9876543325",
        "Type": "zudio"
    },
    {
        "Store Name": "Zudio Amanora Hadapsar",
        "Location": "Amanora Mall, Hadapsar",
        "City": "Pune",
        "Store Manager Name": "Rekha Jadhav",
        "Store Contact Number": "+91-9876543326",
        "Type": "zudio"
    },
    {
        "Store Name": "Zudio Seasons Magarpatta",
        "Location": "Seasons Mall, Magarpatta",
        "City": "Pune",
        "Store Manager Name": "Amit Kadam",
        "Store Contact Number": "+91-9876543327",
        "Type": "zudio"
    },
    {
        "Store Name": "Zudio Elpro Chinchwad",
        "Location": "Elpro City Square, Chinchwad",
        "City": "Pune",
        "Store Manager Name": "Gayatri Pawar",
        "Store Contact Number": "+91-9876543328",
        "Type": "zudio"
    },
    {
        "Store Name": "Zudio Kumar Pacific",
        "Location": "Kumar Pacific Mall, Swargate",
        "City": "Pune",
        "Store Manager Name": "Santosh Gaikwad",
        "Store Contact Number": "+91-9876543329",
        "Type": "zudio"
    }
]

# Create DataFrame
stores_df = pd.DataFrame(stores_data)

# Function to get stores by type
def get_stores_by_type(store_type):
    """Get stores filtered by type (westside or zudio)"""
    return stores_df[stores_df['Type'] == store_type]['Store Name'].tolist()

# Function to get all store names
def get_all_store_names():
    """Get all store names as a list"""
    return stores_df['Store Name'].tolist()

# Function to get store details by name
def get_store_details(store_name):
    """Get complete store details by store name"""
    return stores_df[stores_df['Store Name'] == store_name].iloc[0].to_dict() if not stores_df[stores_df['Store Name'] == store_name].empty else None
