while True:
num1 = input("첫째 숫자를 입력: ")
num1 = int(num1)

num2 = input("둘째 숫자를 입력: ")
num2 = int(num2)

sim = input("연산자를 넣으세요.")

if sim == "+":
    print(num1+num2)
elif sim == "-":
    print(num1-num2)
elif sim == "*":
    print(num1*num2)
else:
    print(num1/num2)

user_input = input(“종료하려면 ‘exit’ 입력: “)
if user_input == “exit”:
break