import argparse

parser = argparse.ArgumentParser(description="Learning Parsing", add_help=True)

parser.add_argument("filename", help="This parameter is to enter a filename")
# help = argparse.SUPRESS --> supresses the help functionality for that argument
parser.add_argument("-c", "--copy", help="Make N copies of the file")
parser.add_argument("-c2", "--copy2", action="store", metavar="N", help="Make N copies of the file")
# action is by default "store"
# Metavar comes in place of copy 2 when displaying the help
parser.add_argument("-s", "--something", action="store_const", const=14, default=12, help="Just tester func")
# this doesn't take any of the values from console.
# If -s is called then it will store 14 and if not called it will store 12
parser.add_argument('-b', '--boolean', action="store_true")
# store_false also there
# stores True if flag called else false
parser.add_argument("-c3", "--copy3", dest="new_dest", help="Make N copies of the file", type=int)
# dest is an attribute that lets the user show copy in help but stores new_dest in the dict
# Accessing the value has to be done with the new_dest
# type specifies the allowed data type
parser.add_argument('-v', '--version', action='version', version="parser 0.1")
parser.add_argument('-n', '--name', choices=['name1', 'name2'])
# gives the user to choose among the choices only
parser.add_argument("-d", "--dopy", nargs='?', default='2', const='3', help="Make N copies of the file")
# n args is optional number of arguments
# default is what comes if flag not called
# $ python testing_p.py -n name1 file.txt
# const is what comes if flag called but not assigned any value
# $ python testing_p.py -n name1 file.txt -d
# if value put after flaqg, that will be considered
# $ python testing_p.py -n name1 file.txt -d 4
parser.add_argument("-d2", "--dopy2", nargs='*', help="Make N copies of the file")
# if putting * here it means that there are many values it can take
# defaults and const cant be here
parser.add_argument("-d3", "--dopy3", nargs='+', help="Make N copies of the file")
# + --> optional
parser.add_argument("-d4", "--dopy4", action="append", help="Make N copies of the file")
# appends all values into a list form
parser.add_argument('-p', '--somethin', action='count', default=0)
# $ python testing_p.py -n name1 file.txt -p -p
# $ python testing_p.py -n name1 file.txt -pp ---- both are same
parser.add_argument('-p2', '--somethin2', action='append', nargs="+")
# this creates list within list which is not ideal
parser.add_argument('-p3', '--somethin3', action='extend', nargs="+")

arguments = parser.parse_args()
print(arguments)
dictt = vars(arguments)
print(dictt)
#
# # ********************************* https://www.youtube.com/watch?v=rJCl7t3IIbA
# import sys
# print(sys.argv)
# arguments = sys.argv[1:]
# if len(arguments) < 1:
#     print(f"Error: At least 1 Argument expected got {len(arguments)}")
#     sys.exit(0)
# arg = arguments[0]
#
#
# def function1():
#     print("First")
#
#
# def function2():
#     print("Second")
#
#
# if arg == '1' or arg == 'one':
#     function1()
# elif arg == '2':
#     function2()
# else:
#     exit(0)