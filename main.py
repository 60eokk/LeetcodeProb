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

        # this is the maximum length of the number that can be created
        res = [0] * (len(num1) + len(num2))

        # go backwards
        for i in range(len(num1) - 1, -1, -1):
            for j in range(len(num2) - 1, -1, -1):
                res[i + j + 1] += int(num1[i]) * int(num2[j])
                res[i + j] += res[i + j + 1] // 10
                res[i + j + 1] %= 10

        i = 0

        while res[i] == 0:
            i += 1

        return "".join(map(str,res[i:]))
    

# Q45
# Greedy algoritm: "making optimal choices for each step"
class Solution:
    def jump(self, nums: List[int]) -> int:
        current, farthest, count = 0,0,0

        for i in range(len(nums)-1):
            farthest = max(farthest, i+nums[i])
            
            if i==current:
                current = farthest
                count+=1
                if current >= len(nums)-1:
                    break
        return count
    

# Q46
# Permutation: every possible lists of a list
# ex: Input: nums = [1,2,3]        Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
# Backtracking: a recursive method that constructs solutions incrementally, abandoning any path as soon as it does not lead to solution
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def backtrack(current_permutation, used):
            # If the current permutation is complete
            if len(current_permutation) == len(nums):
                results.append(current_permutation[:])  # Append a copy of the current permutation
                return
            
            # Try to use each number as the next perm element
            for i in range(len(nums)):
                if not used[i]:  # If this element has not been used
                    used[i] = True
                    current_permutation.append(nums[i])
                    backtrack(current_permutation, used)
                    current_permutation.pop()  # Remove the element added last
                    used[i] = False  # Mark this element as not used
        
        results = []
        backtrack([], [False] * len(nums))  # Initialize used-array with False
        return results
    

# Q47
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        nums.sort()  # Sort the numbers to handle duplicates easily
        results = []  # This will store all unique permutations
        
        def backtrack(current_permutation, used):
            if len(current_permutation) == len(nums):
                results.append(current_permutation[:])  # Append a copy of the current permutation
                return
            
            for i in range(len(nums)):
                if not used[i]:  # Ensure this element hasn't been used in the current permutation
                    # Skip the element if it's the same as the one before it and the previous one wasn't used
                    if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                        continue
                    
                    used[i] = True
                    current_permutation.append(nums[i])
                    backtrack(current_permutation, used)
                    current_permutation.pop()  # Backtrack by removing the last element
                    used[i] = False  # Mark this element as not used

        backtrack([], [False] * len(nums))  # Initialize the used array with False
        return results
    

# Q48
# simple algorithm for "matrix" probs
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        l=0
        r=len(matrix) - 1

        while l<r:
            matrix[l], matrix[r] = matrix[r], matrix[l]
            l += 1
            r -= 1

        # transpose
        for i in range(len(matrix)):
            for j in range(i):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]


# Q49
# Anagram: "word formed by rearranging the letters of a different word
# typically using all the original letters exactly once"
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        sortedtable = {}

        for string in strs:
            sortedstr = 'hi'.join(sorted(string))

            if sortedstr not in sortedtable:
                sortedtable[sortedstr] = []

            sortedtable[sortedstr].append(string)

        return list(sortedtable.values())
    

# Q50
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n==0:
            return 1
        elif n > 0:
            orig = x
            for _ in range(n-1):
                x *= orig
        else:
            orig = x
            for _ in range(-n-1):
                x *= orig
            x = 1/x

        
        return x
# above is correct, but BAD TIME COMPLEXITY
# Thus, better way to solve is to divide into "HALVES"
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n == 0:
            return 1  # Any number to the power of 0 is 1.
        elif n < 0:
            x = 1 / x  # Handle negative powers by taking the reciprocal.
            n = -n

        def helper(x, n):
            if n == 1:
                return x
            half = helper(x, n // 2)
            if n % 2 == 0:
                return half * half
            else:
                return half * half * x

        return helper(x, n)
    
# Q53
# Subarray:" A subarray is a contiguous non-empty sequence of elements within an array"
# Think this could be used for..??
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        if not nums:
            return 0
        currentmax = globalmax = nums[0]
        for i in range(1,len(nums)):
            currentmax = max(nums[i], currentmax + nums[i])

            if currentmax > globalmax:
                globalmax = currentmax

        return globalmax
    
# Q54 
# Wondering where can spiral maxtrix queestiones be used for?
# they can be used in animations, data visualization, games
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if not matrix or not matrix[0]:
            return []
        
        res = []
        left, right = 0, len(matrix[0]) - 1
        top, bottom = 0, len(matrix) - 1
        
        while left <= right and top <= bottom:
            # Traverse from left to right along the top row
            for i in range(left, right + 1):
                res.append(matrix[top][i])
            top += 1  # Move the top boundary downwards AFTER completing the row
            
            # Traverse downwards along the right column
            if top <= bottom:
                for i in range(top, bottom + 1):
                    res.append(matrix[i][right])
                right -= 1  # Move the right boundary to the left
            
            # Traverse from right to left along the bottom row
            if left <= right and top <= bottom:
                for i in range(right, left - 1, -1):
                    res.append(matrix[bottom][i])
                bottom -= 1  # Move the bottom boundary upwards
            
            # Traverse upwards along the left column
            if top <= bottom and left <= right:
                for i in range(bottom, top - 1, -1):
                    res.append(matrix[i][left])
                left += 1  # Move the left boundary to the right
        
        return res
    
# Q55
# thinking a question to another question
# meaning, this questino is going to transform into a function where there is a "car" and it 
# uses up 1 unit of gas everytime it moves and "resets" its gas when it goes to the new position
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        gas = 0
        for n in nums:
            if gas<0:
                return False
            elif n > gas:
                gas = n
            gas -= 1
        return True
    

# Q56
# lamda function is in this format --> lambda arguments: expression
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if not intervals:
            return []

        intervals.sort(key = lambda x: x[0])
        merged = [intervals[0]]
        for i in range(1,len(intervals)):
            if merged[-1][1] >= intervals[i][0]:
                merged[-1][1] = max(merged[-1][1], intervals[i][1])
            else:
                merged.append(intervals[i])

        return merged


# Q57
# By searching for YouTube videos of leetcode solutions, I have realized the importance of "hashsets", 
# and "binary search". This prob focuses on binary search as it hints.
# Binary search allows complexity reduction from O(n) to O(logn)
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        result = []
        n = len(intervals)
        i = 0
        
        # Use binary search to find the correct insertion point
        low, high = 0, n - 1
        while low <= high:
            mid = (low + high) // 2
            if intervals[mid][0] < newInterval[0]:
                low = mid + 1
            else:
                high = mid - 1
        
        # Now 'low' is the correct index to consider for insertion
        # Add all intervals before 'newInterval'
        while i < low:
            result.append(intervals[i])
            i += 1
        
        # Merge 'newInterval' with intervals in 'result' if it overlaps
        if not result or result[-1][1] < newInterval[0]:
            result.append(newInterval)
        else:
            result[-1][1] = max(result[-1][1], newInterval[1])
        
        # Process the rest of the intervals
        while i < n:
            interval = intervals[i]
            if result[-1][1] < interval[0]:
                result.append(interval)
            else:
                result[-1][1] = max(result[-1][1], interval[1])
            i += 1

        return result
    
# Q58
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        wordlist = s.split()
        if wordlist:
            return len(wordlist[-1])
        
# Q59 
# anoter spiral matrix problem (conceptually easy to understand with 4 pointers)
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        matrix = [[0]*n for _ in range(n)] # easy list comprehension to make 2d matrix
        left, right = 0, n-1
        top, bottom = 0, n-1
        val = 1

        while left <= right:
            # fill every val in top row
            for col in range(left, right+1):
                matrix[top][col] = val
                val+=1
            top += 1

            # fill every val in right col
            for row in range(top, bottom+1):
                matrix[row][right] = val
                val += 1
            right -= 1

            # fill every val in bottom row (reverse order)
            for col in range(right, left-1, -1):
                matrix[bottom][col] = val
                val += 1
            bottom -= 1

            # fill every val in left col (reverse order)
            for row in range(bottom, top-1, -1):
                matrix[row][left] = val
                val += 1
            left += 1

        return matrix
    

# Q61
# linked list prob again --> thought process of this q: 
# count backwards k and that is where we want to start. Thus, when looking backwards, the last pointer 
# before should point to NULL and tail (original last pointer) should point back to beginning
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if not head: # if empty
            return head
        
        length, tail = 1, head
        while tail.next:
            tail = tail.next
            length +=1

        k = k%length
        if k==0:
            return head
        
        cur = head
        for i in range(length-k-1):
            cur = cur.next
        newHead = cur.next
        cur.next = None
        tail.next = head
        return newHead


# Q62
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        if m==1 or n==1:
            return 1
        else:
            return int(factorial(m+n-2)/(factorial(n-1)*factorial(m-1)))
# solve by using math


# Q63
# same question with Q62 with an obstacle
# Solve by using brute force OR dynamic programming
# First solution is using dfs (HASHMAP)
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        M, N = len(obstacleGrid), len(obstacleGrid[0])
        dp = {(M-1, N-1): 1}

        def dfs(r,c):
            if r==M or c==N or grid[r][c]:
                return 0
            if (r,c) in dp:
                return dp[(r,c)]
            dp[(r,c)] = dfs(r+1,c) + dfs(r,c+1)
            return dp[(r,c)]
        
        return dfs(0,0)

# Second solution uses dynamic programming (= adding up numbers from bottom right to top left)
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        M, N = len(obstacleGrid), len(obstacleGrid[0])
        dp = [0] * N
        dp[N-1] = 1

        for r in reversed(range(M)):
            for c in reversed(range(N)):
                if obstacleGrid[r][c]:
                    dp[c] = 0
                elif c+1 < N:
                    dp[c] = dp[c] + dp[c+1]
                else:
                    dp[c] = dp[c] + 0
        
        return dp[0]


# Q64
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        n=len(grid)
        m=len(grid[0])
        for i in range(n):
            for j in range(m):
                if i==0:
                    if j!=0:
                        grid[i][j]+=grid[i][j-1]
                elif j==0:
                    if i!=0:
                        grid[i][j]+=grid[i-1][j]
                else:
                    grid[i][j]+=min(grid[i-1][j],grid[i][j-1])
        return grid[n-1][m-1]
    
# Starting to realize that every problem, whether it be leetcode or any other problem
# is very dependent on HOW I look at the problem and solve it
# You can solve it anyway. But need to rethink to use the best, most efficient way

# Q66
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        if digits[-1] != 9:
            digits[-1] += 1
            return digits
        elif len(digits)==1 and digits[0] == 9:
            return [1,0]
        else: 
            digits[-1]=0
            digits[0:-1] = self.plusOne(digits[0:-1])
            return digits
# recursive method