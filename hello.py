all_accounts = []

def cowdec(fn):
    def defination(*args, **kargs):
        for i in all_accounts:
            if i["name"] == args[0] and i["amount"] > 1000:
                return fn(*args, **kargs)
        else:
            return "User have less than 1000 balance"
    return  defination

def createAccount(name, amount):
    all_accounts.append({"name" : name, "amount" : amount})

@cowdec
def getBalance(name):
    for i in all_accounts:
        if i["name"] == name:
            return i["amount"]
    else:
        return "User not found"
    
def addAmount(name, amount):
    for i in all_accounts:
        if i["name"] == name:
            i["amount"] += amount
    else:
        return "User not found"
    
createAccount("Pawan", 100)
createAccount("Inogen", 5000)
print("Pawan", getBalance("Pawan"))
print("Inogen", getBalance("Inogen"))
addAmount("Pawan", 200)
addAmount("Inogen", 1000)
print("Pawan", getBalance("Pawan"))
print("Inogen", getBalance("Inogen"))
