#1
class Solution:
    def twoSum(self, nums:List[int], target:int) -> List[int]:
        n = len(nums)
        for i in range(n):
            for j in range(i+1, n):
                if nums[i] + nums[j] == target:
                    return [i,j]

        return []




# 2
# Definition for singly-linked list.
class ListNode: # represents singly linked list
    # Each node contains int value stored in the node + a pointer do the next node ('next'). If no next node, then 'None'
    def __init__(self, val=0, next=None):
        # __init__ is a constructor method for a  new instance of ListNode
        self.val = val
        self.next = next


class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        return 0




# 3
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:


        seen = set()
        max_num = 0
        i=0
        j=0

        while j < len(s):
            if s[j] not in seen:
                seen.add(s[j])
                j+=1
                max_num = max(max_num, j-i)
            else:
                while s[i] != s[j]:
                    seen.remove(s[i])
                    i+=1
                seen.remove(s[i])
                i+=1

        return max_num
        


# 5
class Solution:
    def longestPalindrome(self, s: str) -> str:
        if len(s) <=1:
            return s

        max_num = 1
        max_str = s[0]
        
        for i in range(len(s)-1):
            for j in range(i+1, len(s)):
                if j-i+1 > max_num and s[i:j+1] == s[i:j+1][::-1]:
                    max_num = j-i+1
                    max_str = s[i:j+1]
        
        return max_str
    

# 6
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        
        if numRows == 1:
            return s
        
        buckets : list[str] = ["" for i in range(numRows)]

        i=0
        multiplier = 1

        for char in s:
            buckets[i] += char

            i += 1 * multiplier
            if i==numRows:
                i -= 2
                multiplier = -1

            if i == -1:
                i += 2
                multiplier = 1

        resultString = ""
        for string in buckets:
            resultString += string

        return resultString
    


# 7
class Solution:
    def reverse(self, x: int) -> int:
        INTMAX = 2**31 - 1
        INTMIN = -2**31

        if x < 0:
            sign = -1
        else:
            sign = 1

        x = x * sign

        reverse = int(str(x)[::-1])

        reverse = reverse * sign

        if reverse < INTMIN or reverse > INTMAX:
            return 0
        else:
            return reverse
        

# Q 8
class Solution:
    def myAtoi(self, s:str) -> int:
        n = len(s)
        i=0
        result = ""
        sign=1

        while i<n and s[i] == " ":
            i += 1
        if i<n and (s[i] == "-" or s[i] == "+"):
            sign = -1 if s[i] == "-" else 1
            i += 1
        while i<n and s[i].isdigit():
            result += s[i]
            i += 1

        if result:
            ans = int(result)
            ans *= sign
        else:
            return 0
        # LINE148-152 IS IMPORTANT: DO NOT CONVERT RESULT TO ANS FIRST. BECAUSE YOU GOTTA CHECK WHETHER RESULT IS EMPTY OR NOT!!
        # IF RESULT WAS EMPTY AND I COVERTED ans = int(result), it would have returned error

        if ans <= -2**31:
            ans = -2**31
        elif ans >= 2**31 - 1:
            ans = 2**31 - 1

        return ans

# Q9
class Solution:
    def isPalindrome(self, x: int) -> bool:
        if len(str(x)) == 1:
            return True
        for i in range(len(str(x))):
            return True if str(x) == str(x)[::-1] else False
        

# Q11
# For efficiency, for loop inside for loop: O(n^2), a while loop is O(n). 
class Solution:
    def maxArea(self, height: List[int]) -> int:

        left = 0
        right = len(height)-1
        maxArea = 0

        
        while left < right:
            currentArea = min(height[left], height[right]) * (right-left)
            maxArea = max(currentArea, maxArea)

            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        
        return maxArea


# Q12
class Solution:
    def intToRoman(self, num: int) -> str:
        romanNumeral = {
            1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII", 9: "IX",
            10: "X", 20: "XX", 30: "XXX", 40: "XL", 50: "L", 60: "LX", 70: "LXX", 80: "LXXX", 90: "XC",
            100: "C", 200: "CC", 300: "CCC", 400: "CD", 500: "D", 600: "DC", 700: "DCC", 800: "DCCC", 900: "CM",
            1000: "M", 2000: "MM", 3000: "MMM"
            }

        res = ""

        if num in romanNumeral:
            return romanNumeral[num]

        else:
            while num:
                divisor = 10** (len(str(num))-1)
                check = divisor * (num//divisor)
                res += romanNumeral[check]
                num = num % divisor
            return res


# Q13
# lambda functions: can be used for calculus function, create function anonymously
def f(x):
    return x**2
f = lambda x: x**2 # they are the same

lambda x,y,z: x*y*z

# lambda is often used with map
L = [2,3,4,5]
m = map(lambda x: x**3, L) 
# map(lambda x: A, B)  --> A is output, B is the variable

F = filter(lambda x: x <=4, L)
print(list(F)) # output: [2,3,4]

c = map(lambda x: x**3, filter(lambda x: x<=4, L)) # nested version of above 2
print(list(c))


# replace function
txt = "ABBBC"
x = txt.replace("BB", "XX")
print(x) # output: AXXBC (meaning it checks from left to right in order)

# GOING BACK TO Q13
class Solution:
    def romanToInt(self, s: str) -> int:
        roman_to_integer = {
            'I': 1,'V': 5, 'X': 10, 'L': 50,'C': 100,'D': 500,'M': 1000,
        }
        s = s.replace("IV", "IIII").replace("IX", "VIIII").replace("XL", "XXXX").replace("XC", "LXXXX").replace("CD", "CCCC").replace("CM", "DCCCC")
        return sum(map(lambda x: roman_to_integer[x], s)) 
    


# Q14
# putting them in sorted (ascending order) makes this question very easy to solve
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        strs = sorted(strs)
        first = strs[0]
        last = strs[-1]
        ans = ""

        for i in range(min(len(first), len(last))):
            if first[i] != last[i]:
                return ans
            ans += first[i]
            
        return ans