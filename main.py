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
    

# Q15
# continue and break are used to control the flows of loops
# continue: skip the rest of the code inside a loop for the current iteration.
# continue: is often used when a specific condition is met, and you want to skip the rest of the loop body without exiting the loop
# example: will only output odd numbers
for i in range(10):
    if i % 2 == 0:
        continue  # Skip the rest of the loop for even numbers
    print(i)  # This line is only executed for odd numbers

# break: exit loop entirely. once break is executed, the loop is terminated. exmaple below
for i in range(10):
    if i == 5:
        break  # Exit the loop when i is 5
    print(i)

# back to question
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        newList = []
        nums.sort()
        
        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            
            j = i+1
            k = len(nums)-1

            while j<k:
                total = nums[i] + nums[j] + nums[k]
                if total > 0:
                    k-=1
                elif total < 0:
                    j+=1
                else:
                    newList.append([nums[i], nums[j], nums[k]])
                    j+=1

                    while nums[j] == nums[j-1] and j<k:
                        j+=1
        return newList


# Q16
# float('inf) means positive infinity, while float('-inf') would be negative infinity
# This question is basically making a "dummy" to compare with the abs(ans-target) and keep updating ans accordingly
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        
        n = len(nums)
        nums.sort()
        ans = float('inf')

        for i in range(n):
            left = i+1
            right = n-1

            while (left < right):
                cur = nums[i] + nums[left] + nums[right]
                if abs(cur-target) < abs(ans-target):
                    ans = cur
                
                if cur < target:
                    left += 1
                elif cur > target:
                    right -= 1
                else:
                    return cur
            
        return ans


# Q17
# queue: FIFO (Handling tasks on printer, request on server, BFS in graph or tree algorithm)
# deque (double ended queue): Will use when you need to add/pop/etc on front / rear of a list
# Thus, deque is more frequently used over queue in lists
## queue.pop() will delete and return the first element
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        phone = {
            '2':['a','b','c'], '3':['d','e','f'], '4':['g','h','i'], '5':['j','k','l'], 
                  '6':['m','n','o'], '7':['p','q','r','s'], '8':['t','u','v'], '9':['w','x','y','z']
        }

        solution = [""]

        for i in digits:
            solution = [x + y for x in solution for y in phone[i]]
        return [] if solution == [""] else solution
    

# Q18
# vector: dynamic array mostly used in C++
# THIS ONE IS HARD! TAKING A WHILE
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        ans = set()
        nums.sort()
        for i in range(len(nums)):
            for j in range(i+1,len(nums)):
                k,l = j+1,len(nums)-1
                while k<l:
                    s = nums[i]+nums[j]+nums[k]+nums[l]
                    if s == target:
                        ans.add((nums[i],nums[j],nums[k],nums[l]))
                        l-=1
                        k+=1
                    elif s > target:
                        l-=1
                    else:
                        k+=1
                  
        return ans
    

# Q19
# ListNode: typically a class used an element (or node) within a singly linked list
# Each node contains: Data(actual value stored inside node) + Next(pointer to next node)
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val  # The data value
        self.next = next  # Reference to the next ListNode in the list


class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy = ListNode(0)
        dummy.next = head
        first = dummy
        second = dummy

        for _ in range(n+1):
            first = first.next

        while first:
            first = first.next
            second = second.next

        second.next = second.next.next
        # In singly linked list, when you want to delete a node, you don't "delete", but you would refer to the next node (exclude that node from the sequance)

        return dummy.next
        # this does not only return the next node, but the whole list after dummy's next pointer which would be 1, thus 1235


#Q20
class Solution:
    def isValid(self, s: str) -> bool:
        while '()' in s or '[]'in s or '{}' in s:
            s = s.replace('()','').replace('[]','').replace('{}','')
        return False if len(s) !=0 else True
    

# Q21
# Iteration is commonly used!
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:   
        if not l1 or not l2:
            return l1 or l2
        
        if l1.val <= l2.val: #1
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else: #2
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2
        

# Q22
# Always think about "backtracking" in questions itself involves some kind of iteration
