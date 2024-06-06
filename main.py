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
class Solution(object):
    def generateParenthesis(self, n):
        def Generate_Parentheses(sentence, left, right):
            if len(sentence) == n * 2:
                l.append(sentence)
                return
            if left < n:
                Generate_Parentheses(sentence + '(', left + 1, right)
            if right < left:
                Generate_Parentheses(sentence + ')', left, right + 1)

        l = []
        Generate_Parentheses('',0,0)
        return l
# it only adds a "right" only if the number of "left" is less than "right" --> which is the correct logic


# Q24
# Node questions are difficult! Do not think like a list. Should think as a "pointer"
# Think of "dummy" when dealing with nodes. 
# Dummy is mostly used as a placeholder before the head of the list to handle edge cases smoothly

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def swapPairs(self, head:ListNode) -> ListNode:
        dummy = ListNode(-1)
        dummy.next = head
        prev_node = dummy

        while prev_node.next and prev_node.next.next:
            node1 = prev_node.next
            node2 = node1.next

            prev_node.next = node2
            node1.next = node2.next
            node2.next = node1
            
            prev_node = node1

        return dummy.next
    

# Q26
# remove duplicates "in-place": means removing duplicates without requiring extra space proportional to input size
# this means that it modifies the input in place without creating a copy of it
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        count = 1
        for i in range(1, len(nums)):
            if nums[i] != nums[i-1]:
                nums[count] = nums[i]
                count +=1
        return count
# import part of this was thinking of nums[i] != nums[i-1]. it is not always i+1


# Q27
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:

        count =0
        for i in range(len(nums)):
            if nums[i] != val:
                nums[count] = nums[i]
                count +=1
        return count

# Q28
# think about "slicing"
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        for i in range(len(haystack)):
            if haystack[i:i+len(needle)] == needle:
                return i
        return -1
    

# 29
class Solution:
    def divide(self, dividend: int, divisor: int) -> int:
        sign = -1 if (dividend >= 0 and divisor < 0) or (dividend < 0 and divisor >= 0) else 1
        dividend = abs(dividend)
        divisor = abs(divisor)
        result = len(range(0, dividend-divisor+1, divisor))
        if sign == -1:
            result = -result
        minus_limit = -(2**31)
        plus_limit = (2**31 - 1)
        result = min(max(result, minus_limit), plus_limit)
        return result
# encountered a lot of time limit exceeded errors. 
# managed to solve by thinking of another algorihtm and keeping the min and max nums


# Q31
# tricky part about this was that the function did not return anything
# instead, change the nums itself, without using any extra spaces
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        
        i = len(nums) -2
        while i>=0 and nums[i] >= nums[i+1]:
            i -=1

        if i==-1:
            nums.reverse()
        else:
            pos = i+1
            for j in range(len(nums)-1, i, -1):
                if nums[j] > nums[i] and nums[j] < nums[pos]:
                    pos =j

            nums[i], nums[pos] = nums[pos], nums[i]
            nums[i+1:] = sorted(nums[i+1:])


# Q33
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        for i in range(len(nums)):
            if nums[i] == target:
                return i
        else:
            return -1
# above solution doesn't meet the requirements of solving in O(logn)
# In order to solve this in O(logn) rutime complexity, use "modified binary search"
# Think of "mid point"--> check left and right to see if it is ordered normally and whether target is in that half
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums)-1

        while left <= right:
            mid = (left+right)//2
            if nums[mid]==target:
                return mid
            
            # check if left half is normally ordered
            if nums[left] <= nums[mid]:
                # target is in left half
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            # right half is normally ordered
            else:
                if nums[mid] < target <= nums[right]:
                    left = mid +1
                else:
                    right =mid-1
        return -1
    
# Q34
# Question is again asking to solve in O(logn) --> think of binary search
class Solution:
    def binarySearch(self, nums, target, side):
        left, right = 0, len(nums) - 1
        index = -1
        while left <= right:
            mid = (left + right) // 2
            if target > nums[mid]:
                left = mid + 1
            elif target < nums[mid]:
                right = mid - 1
            else:
                index = mid
                if side == 0:
                    right = mid - 1
                else:
                    left = mid + 1
        return index

    def searchRange(self, nums: List[int], target: int) -> List[int]:
        # '0' for left side, '1' for right side
        left = self.binarySearch(nums, target, 0)
        right = self.binarySearch(nums, target, 1)
        return [left, right]
    
# ANOTHER SOLUTION (EASIER TO COMPREHEND):
class Solution:
    def findFirst(self, nums, target):
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] < target:
                left = mid + 1
            elif nums[mid] > target:
                right = mid - 1
            else:
                if mid == 0 or nums[mid - 1] != target:
                    return mid
                right = mid - 1
        return -1

    def findLast(self, nums, target):
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] < target:
                left = mid + 1
            elif nums[mid] > target:
                right = mid - 1
            else:
                if mid == len(nums) - 1 or nums[mid + 1] != target:
                    return mid
                left = mid + 1
        return -1

    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if not nums:
            return [-1, -1]
        
        first = self.findFirst(nums, target)
        if first == -1:
            return [-1, -1]
        last = self.findLast(nums, target)
        
        return [first, last]
    

# Q35
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums)-1
       
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
            
        return 0
# above is the solution without the condition of "If not, return the index where it would be if it were inserted in order."
# below is the full solution
# Just have to change return 0 --> return left (the insertion point)

# Q36
# set() will remove duplicates
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        res = []
        for i in range(9):
            for j in range(9):
                element = board[i][j]
                if element != ".":
                    res += [(i,element), (element, j), (i//3, j//3, element)]
        return len(res)==len(set(res))
    

# Q38
# Run-length encoding: form of lossless data compression in which runs of data (consecutive occurrences of the same data value) 
# are stored as a single occurrence of that data value and a count of its consecutive occurrences, rather than as the original run
# green green green would be green x3
class Solution:
    def countAndSay(self, n: int) -> str:
        if n==1:
            return "1"
        previous = self.countAndSay(n-1)

        return self.generateNext(previous)

    def generateNext(self,s):
        result = []
        count = 1
        
        for i in range(1,len(s)):
            if s[i] ==s[i-1]:
                count+=1
            else:
                result.append(str(count) + s[i-1])
                count=1

        result.append(str(count) + s[-1])

        return ''.join(result)
    

# break, continue
# "break" will INSTANTLY EXIT the loop
# "continue" will SKIP and move onto NEXT loop
numbers = [1, 2, 3, 4, 5, -1, 7, 8]
for number in numbers:
    if number == -1:
        print("Skipping negative number:", number)
        continue  # Skip this iteration
    print("Processing number:", number)

# above will output the following: so if i dont want it to exit and keep going inside the loop i will use "continue"
#Processing number: 1
#Processing number: 2
#Processing number: 3
#Processing number: 4
#Processing number: 5
#Skipping negative number: -1
#Processing number: 7
#Processing number: 8


# Q39
# Backtrack: technique for solving recursively by trying to build a solution incrementally
# DFS: Depth firth Search: explores possible vertices (from a given starting point) down each branch before backtracking
# This property of DFS allows algorithm to dive deep into a tree/graph using one path first, then it backtracks and explores next path
# Backtrack solution below
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        def backtrack(start, comb, target):
            if target ==0: # current combination comb is a valid solution
                res.append(list(comb))
                return
            if target<0: # means we exceeded target sum with current combination
                return

            # iterate over candidates starting from 'start' index 
            for i in range(start, len(candidates)):
                comb.append(candidates[i])
                # pass i as start again because we can use duplicates
                backtrack(i,comb,target-candidates[i])
                comb.pop()

        res = []

        backtrack(0,[], target)
        return res
    
# DFS Solution below:
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        def dfs(start, path, remaining):
            if remaining == 0:
                # Found a combination whose sum is target
                results.append(list(path))
                return
            if remaining < 0:
                # Exceeded the sum, no need to proceed further
                return

            for i in range(start, len(candidates)):
                # Include the current number and move deeper into the DFS
                path.append(candidates[i])
                dfs(i, path, remaining - candidates[i])
                # Backtrack: remove the last added number and try the next possible number
                path.pop()

        results = []
        dfs(0, [], target)
        return results
    
# Q40
# utilizing previous question + contineu/break
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()  # Sort candidates to help skip duplicates
        results = []

        def backtrack(comb, remaining, start):
            if remaining == 0:
                results.append(list(comb))  # Found a valid combination
                return
            for i in range(start, len(candidates)):
                # Skip duplicates
                if i > start and candidates[i] == candidates[i - 1]:
                    continue
                if remaining - candidates[i] < 0:
                    break  # No need to continue if the remaining sum becomes negative
                # Include the candidate and move to the next
                comb.append(candidates[i])
                backtrack(comb, remaining - candidates[i], i + 1)  # Move to the next index
                comb.pop()  # Backtrack

        backtrack([], target, 0)
        return results
    
# Q43
# again, another problem to think closely about the algorithm
# the way so solve it is reversing num1, num2 backwards and multiplying the rightmost digits first and going left
class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        if num1 == '0' or num2 == '0':
            return '0'

        res = [0] * (len(num1) + len(num2))

        for i in range(len(num1) - 1, -1, -1):
            for j in range(len(num2) - 1, -1, -1):
                res[i + j + 1] += int(num1[i]) * int(num2[j])
                res[i + j] += res[i + j + 1] // 10
                res[i + j + 1] %= 10

        i = 0

        while res[i] == 0:
            i += 1

        return "".join(map(str,res[i:]))