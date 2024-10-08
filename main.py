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


# Q67
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        a,b = a[::-1], b[::-1]
        res = []
        carry = 0

        for i in range(max(len(a), len(b))):
            x= int(a[i]) if i <len(a) else 0 # good way for "if index out of range, treat as 0"
            y= int(b[i]) if i <len(b) else 0
            total = x+y+carry

            carry = total//2
            res.append(total%2)
        
        if carry:
            res.append(carry)
        return ''.join(str(x) for x in res[::-1])

# Q69
class Solution:
    def mySqrt(self, x: int) -> int:
        left, right = 1, x
        while left<=right:
            mid = (left+right)//2

            if mid*mid == x:
                return mid
            elif mid*mid <x:
                left = mid+1
            else:
                right = mid-1
        return right
# easy binary search prob


# Q70
# dynamic programming is very much related to memoization (faster)
# Example of recursive VS memoization
# The simple way to define them recursively in Sage is as follows:
def fib(n):
    # Base Cases n = 0 or n = 1
    if n == 0:
        return 0
    if n == 1:
        return 1
    # Recursive step
    return fib(n - 1) + fib(n - 2)

def fib_memo(n, D = {}):
    # Check if its already in the dictionary:
    if n in D:
        return D[n]
    else:
        # Base cases
        if n == 0:
            out = 0
        elif n == 1:
            out = 1
        else:
            # Recursive step with memoization
            out = fib_memo(n-1) + fib_memo(n-2)
            D[n] = out
        return out
# Solution for Q70
class Solution:
    def climbStairs(self, n: int) -> int:
        memo = [-1] * (n + 1)
        
        def climb(i):
            if i <= 1:
                return 1
            if memo[i] != -1:
                return memo[i]
            memo[i] = climb(i - 1) + climb(i - 2)
            return memo[i]

        return climb(n)
        

# Queue, Deque
# It is more efficient to import "deque" instead of popping from the front because
# that would actually be O(n), not O(1).
# Thus it is more efficient to import deque like below
from collections import deque
people = ['Mario', 'Luigi', 'Toad']
queue = deque(people)
queue.append('Bowser') # this will append Bowser to the end
queue.popleft # this will pop Mario with O(1)
queue.appendleft('Daisy')
queue.rotate(-2) # This will move everyone to the left 2 positions
queue.reverse() # This will reverse: meaning right to left


# Q71
# Perfect prob to use Stack
class Solution:
    def simplifyPath(self, path: str) -> str:
        stack = []
        cur = ""

        for c in path + "/":
            if c=="/":
                if cur == "..":
                    if stack: # if stack is not empty
                        stack.pop()
                elif cur != "" and cur != ".":
                    stack.append(cur)
                cur = ""
            else:
                cur += c
        return "/" + "/".join(stack)

# Q72
# 2d dynamic programming prob
# thought: think of all ways letters can function(add, delete, replace)
# It is solved by "keep dividing letters into subletters", where each cell cache[i] will hold the 
# minimum num of operations

class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        cache = [[float("inf")] * (len(word2) + 1) for i in range(len(word1) + 1)]

        for j in range(len(word2) + 1):
            cache[len(word1)][j] = len(word2) - j
        for i in range(len(word1) + 1):
            cache[i][len(word2)] = len(word1) - i

        for i in range(len(word1) - 1, -1, -1):
            for j in range(len(word2) - 1, -1, -1):
                if word1[i] == word2[j]:
                    cache[i][j] = cache[i + 1][j + 1]
                else:
                    cache[i][j] = 1 + min(cache[i + 1][j], cache[i][j + 1], cache[i + 1][j + 1])

        return cache[0][0]


# Q73
# Should we make a COPY? If not, it will keep replacing 0 everywhere because it will keep updating
# But the above way is O(m*n): inefficient
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        # O(1)
        ROWS, COLS = len(matrix), len(matrix[0])
        rowZero = False

        # Determine which rows/cols need to be zero
        for r in range(ROWS):
            for c in range(COLS):
                if matrix[r][c] == 0:
                    matrix[0][c] = 0
                    if r > 0:
                        matrix[r][0] = 0
                    else:
                        rowZero = True

        for r in range(1, ROWS):
            for c in range(1, COLS):
                if matrix[0][c] == 0 or matrix[r][0] == 0:
                    matrix[r][c] = 0

        if matrix[0][0] == 0:
            for r in range(ROWS):
                matrix[r][0] = 0

        if rowZero:
            for c in range(COLS):
                matrix[0][c] = 0


# Q74
# Questions makes us solve in O(logm*n). When I see log--? I think of binary
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
            ROWS, COLS = len(matrix), len(matrix[0])
            top, bot = 0, ROWS-1
            while top <= bot:
                row = (top+bot)//2
                if target > matrix[row][-1]:
                    top = row+1
                elif target < matrix[row][0]:
                    bot = row-1
                else:
                    break

            if not (top<=bot):
                return False
            row = (top+bot)//2
            l,r = 0, COLS-1

            while l<=r:
                m = (l+r)//2
                if target > matrix[row][m]:
                    l = m+1
                elif target < matrix[row][m]:
                    r = m-1
                else:
                    return True
            return False
    
# Q75
# mergesort, quicksort: nlog(n)
# bucketsort: linear time O(n) because values are only 0,1,2
# Quicksort with Partition. EX: if there is 1,2,6,7 and the way to divide is n<5, then if 1,2 are put, the rest are on the other side by default
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        zeros, ones, n = 0, 0, len(nums)
        
        # Count the number of 0s and 1s
        for num in nums:
            if num == 0:
                zeros += 1
            elif num == 1:
                ones += 1

        # Fill the array with 0s for the first 'zeros' positions
        for i in range(0, zeros):
            nums[i] = 0

        # Fill the array with 1s for the next 'ones' positions
        for i in range(zeros, zeros + ones):
            nums[i] = 1

        # Fill the array with 2s for the remaining positions
        for i in range(zeros + ones, n):
            nums[i] = 2

# Q77
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        res = []

        def backtrack(start, comb):
            if len(comb)==k:
                res.append(comb.copy()) # copy is used so numbers aren't duplictaed
                return
            
            for i in range(start, n+1):
                comb.append(i)
                backtrack(i+1, comb)
                comb.pop()

        backtrack(1, [])
        return res

# Q78
# For every number, there are 2 choice to include / or not include
# Making a binary tree like going down from top to bottom
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = []
        subset = []

        def dfs(i):
            if i >= len(nums):
                res.append(subset.copy())
                return 
            
            # decision to include
            subset.append(nums[i])
            dfs(i+1)

            # decision to NOT include
            subset.pop()
            dfs(i+1)

        dfs(0)
        return res

# Q79
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        rows, cols = len(board), len(board[0])
        path = set() # so that I don't revisit same char twice

        def dfs(r,c, i):
            if i==len(word):
                return True
            if r<0 or c<0 or r>=rows or c>=cols or word[i]!= board[r][c] or (r,c) in path:
                return False
            
            path.add((r,c))
            res = dfs(r+1,c,i+1) or dfs(r-1,c,i+1) or dfs(r,c+1,i+1) or dfs(r,c-1,i+1)
            path.remove((r,c))
            return res

        for r in range(rows):
            for c in range(cols):
                if dfs(r,c,0): return True

        return False
# sometimes like this solution above, the most efficient way to solve aproblem might be brute force

# Q80
# The problem has a constraint of "You must do this by modifying the input array in-place with O(1) extra memory"
# This makes us solve this problem with "two pointer"
# 1 pointer for iterating thru array, 1 pointer to keep track of where to put next element
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if len(nums) < 3:
            return len(nums)
        
        # Start writing from the third item
        write_index = 2
        
        # Iterate over the array starting from the third element
        for i in range(2, len(nums)):
            # Only write the item to write_index if it's not a duplicate
            # exceeding the allowed two occurrences
            if nums[i] != nums[write_index - 2]:
                nums[write_index] = nums[i]
                write_index += 1
        
        return write_index

# Q81
# Another binary search problem. Which wil be solved l, r and //2, etc
# if nums[L] < nums[M] M(middle), then we are on left side, others, on right side
# Originally thought of the above method, but BECAUSE there is duplicates, it cannot be solved that way
class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        l, r = 0, len(nums)-1
        while l<=r:
            m = l+ (r-l)//2
            if nums[m]  == target:
                return True
            if nums[m] > nums[l]: # left portion
                if nums[m] > target >= nums[l]:
                    r= m-1
                else:
                    l = m+1
            elif nums[m] < nums[l]: # right portion
                if nums[m] < target <= nums[r]:
                    l = m+1
                else:
                    r = m-1
            else:
                l += 1

        return False

# Q82
# Logic for this prob would be to think of cur.next and cur.next.next (will they be the same?)
# For these linked lists questions, always think of creating a dummy
class Solution:
    def deleteDuplicates(self, head:ListNode) -> ListNode:
        dummy = ListNode(0)
        dummy.next = head

        prev = dummy # will be used to connect non-duplicate nodes
        current = head # explore the list

        while current:
            if current.next and current.val == current.next.val:
                while current.next and current.val == current.next.val:
                    current = current.next
                prev.next = current.next # skip duplicates
            else:
                prev = current
            current = current.next
        
        return dummy.next


# Q83
# Very similar from problem above
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
# practice writing the ListNode func (singly linked list)
class Solution:
    def deleteDuplicates(self, head):
        current = head

        while current and current.next:
            if current.val == current.next.val:
                current.next = current.next.next
            else:
                current = current.next

        return head
    

# Q86
# The way: create 2 sublists (one less than val, one bigger or equal to val)
def partition(self, head, x):
    left, right = ListNode(), ListNode()
    ltail, rtail = left, right

    while head:
        if head.val < x:
            ltail.next = head
            ltail = ltail.next
        else:
            rtail.next = head
            rtail = rtail.next
        head.next = head

    ltail.next = right.next # right only by itself is a dummy node, so right.next will point to actual first node
    rtail.next = None
    return left.next


# Q88
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        last = m+n-1 # last index
        
        # merge in reverse order
        while m>0 and n>0:
            if nums1[m-1] > nums2[n-1]:
                nums1[last] = nums1[m-1]
                m -=1
            else:
                nums1[last] = nums2[n-1]
                n -=1
            last -=1

        while n>0:
            nums1[last] = nums2[n-1]
            n -=1
            last -=1


# Q89
# interesting question: feel like this would be another prob to think creatively
class Solution:
    def grayCode(self, n: int) -> List[int]:
        
        result = [i^(i//2) for i in range(pow(2,n))]

        return result


# Q90
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        res = []
        nums.sort()

        def dfs(index, path):
            res.append(path)
            for i in range(index, len(nums)):
                if i > index and nums[i] == nums[i-1]:
                    continue
                dfs(i+1, path+[nums[i]])
        dfs(0, [])
        return res
    
# Q91
# A good way would be to divide into trees (doesn't have to be perfect tree), basically dividing into small sub probs
class Solution:
    # recursive caching solution
    def numDecodings(self, s):
        dp = { len(s) : 1}
        def dfs(i):
            if i in dp:
                return dp[i]
            if s[i] == "0":
                return 0
            
            res = dfs(i+1)

            if (i+1 < len(s) and (s[i] == "1" or (s[i] == "2" and s[i+1] in "0123456"))):
                res += dfs(i+2)
            dp[i] = res
            return res
        return dfs(0)


# Q92
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        dummy = ListNode(0, head)

        # 1) reach node at position "left"
        leftPrev, cur = dummy, head
        for i in range(left - 1):
            leftPrev, cur = cur, cur.next

        # Now cur="left", leftPrev="node before left"
        # 2) reverse from left to right
        prev = None
        for i in range(right - left + 1):
            tmpNext = cur.next
            cur.next = prev
            prev, cur = cur, tmpNext

        # 3) Update pointers
        leftPrev.next.next = cur    # cur is node after "right"
        leftPrev.next = prev        # prev is "right"

        return dummy.next
    

# Q93
class Solution:
    def resotreIPAddtresses(self, s):
        res = [] # this is going to be appended
        # valid: 0-255 inclusive, no leading zeros
        # return list of valid IP
        # another prob of thinking of "diving into subprobs like a tree"
        if len(s) > 12:
            return res
        
        def backtrack(i, dots, curIP): #index, current dots (total 4), currentIP
            if dots == 4 and len(s):
                res.append(curIP[:-1]) # remove the last dot, everything up until -1 index
                return # return without anything this does not return anything and exits
            if dots>4:
                return
            
            for j in range(i, min(i+3, len(s))): # use min because i+3 might be out of range
                if int(s[i:j+1]) < 256 and (i==j or s[i] != 0): # second condition is for "non-leading zeros"
                    backtrack(j+1, dots+1, curIP+s[i:j+1] + ".")

        backtrack(0,0,"")
        return res
    
# Q94
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res = []

        def inorder(root):
            if not root:
                return
            inorder(root.left)
            res.append(root.val)
            inorder(root.right)

        inorder(root)
        return res
    
# but there is another sol: using Stack (iterative)
class Solution:
    def inorderTraversal2(self, root):
        res = []
        stack = []
        cur = root

        while cur or stack:
            while cur:
                stack.append(cur)
                cur = cur.left
            cur = stack.pop()
            res.append(cur.val)
            cur = cur.right
        return res
    

# Q95
# Binary Search Tree: "All values greater has to go on the right side"
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def generateTrees(self, n: int) -> List[Optional[TreeNode]]:
        def generate(left, right):
            if left==right:
                return [TreeNode(left)] # wrap around an array
            if left>right:
                return [None] # put a NULL so it doesn't even not return anything
            res = []
            
            for val in range(left, right+1):
                for leftTree in generate(left, val-1):
                    for rightTree in generate(val+1, right):
                        root = TreeNode(val, leftTree, rightTree)
                        res.append(root)
            return res
        return generate(1,n)
    

# Q96
# Trees again, which means this is most likely going to be solved by subprobs
# Problem is only asking for a number, which means it is added up by multiplying up subprobs
class Solution:
    def numTrees(self, n):
        numTree = [1] * (n+1)
        for nodes in range(2, n+1):
            total = 0
            for root in range(1, nodes+1):
                left = root-1
                right = nodes-root
                total += numTree[left] * numTree[right]
            numTree[nodes] = total
        return numTree[n]

# Q97
# Divide into subprobs: first check whether it will start from s1 or s2
# After that, it is another subprob of which string it will start from
# We can use dynamic programming, by building a 2d grid and ADDING ADDITIONAL LAYER
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        if len(s1) + len(s2) != len(s3):
            return False

        dp = [[False] * (len(s2) + 1) for i in range(len(s1) + 1)]
        dp[len(s1)][len(s2)] = True

        for i in range(len(s1), -1, -1):
            for j in range(len(s2), -1, -1):
                if i < len(s1) and s1[i] == s3[i + j] and dp[i + 1][j]:
                    dp[i][j] = True
                if j < len(s2) and s2[j] == s3[i + j] and dp[i][j + 1]:
                    dp[i][j] = True

        return dp[0][0]
    
# Q98
# The point of this problem is thinking which values to "compare", as we know that the node is between -inf, inf
# When pointer goes down left, it will need to switch the upper bound to node.val, and when going right, it needs to switch lower bound to node.val
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def helper(node, left, right):
            if not node:
                return True

            if not (node.val < right and node.val > left):
                return False

            # now the recursive part
            return (helper(node.left, left, node.val) and 
            helper(node.right, node.val, right))

        return helper(root, float("-inf"), float("inf"))
    
# Q99
# The method is to keep track of the "nodes", instead of the "values" and in the end swap the value
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorder(self, root, li):
        if not root:
            return li
        li = self.inorder(root.left, li)
        li.append(root)
        li = self.inorder(root.right, li)
        return li
    def recoverTree(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        li = self.inorder(root, [])
        n = len(li)
        i,j = 1, n-2
        a = li[0]
        for i in range(1,n):
            if li[i].val < li[i-1].val:
                a = li[i-1]
                break
        b = li[-1]
        for i in range(n-2,-1,-1):
            if li[i].val > li[i+1].val:
                b = li[i+1]
                break
        a.val, b.val = b.val, a.val

# Q100
# YAY! 100 Questions Solved!
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        
        if not p and not q:
            return True
        
        if (not p and q) or (p and not q):
            return False

        if p.val != q.val:
            return False

        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
    
# Q101
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        def dfs(left, right):
            if not left and not right:
                return True
            if not left or not right:
                return False
            
            return (left.val == right.val and
            dfs(left.left, right.right) and
            dfs(left.right, right.left))

        return dfs(root.left, root.right)
    

# Q102
# Traverse level order: Breadst First Search!! (BFS)
# Queue: First in First OUt (FIFO): Basically for this prob: Add by levels and then pop for each level from the queue

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        res = []
         # res.append()

        q = collections.deque()
        q.append(root)

        while q:
            qlength = len(q)
            level = []
            for i in range(qlength):
                node = q.popleft()

                if node:
                    level.append(node.val)
                    q.append(node.left)
                    q.append(node.right)
            if level:
                res.append(level)

        return res
    
# Q103
# thought process: basically this prob is meant to return nodes of each level (zigzag order)
# So for even number level: l -> r, odd number: r -> l
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:

        level = 0
        res = []

        if not root:
            return []
        
        q = collections.deque([root])

        while q:
            level_nodes = []
            for _ in range(len(q)):
                curr = q.popleft()
                if level % 2 == 0:
                    level_nodes.append(curr.val)
                else:
                    level_nodes.insert(0,curr.val)
                if curr.left:
                    q.append(curr.left)
                if curr.right:
                    q.append(curr.right)
            res.append(level_nodes)
            level+=1
        
        return res


# Q104
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if root == None: return 0

        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1


# Q105
# Preorer: Nodes are visited in order (Root - Left - Right)
# Inorder: Traverse left subtree inorder, then root, then right subtree inorder

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if not preorder or not inorder:
            return None
        
        root=TreeNode(preorder[0])
        index=inorder.index(preorder[0])
        root.left=self.buildTree(preorder[1:index+1],inorder[:index])
        root.right=self.buildTree(preorder[index+1:],inorder[index+1:])
        
        return root
    
# Q106
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        if not inorder or not postorder:
            return None

        # The last element in postorder is the root of the current tree
        root_val = postorder.pop()
        root = TreeNode(root_val)

        # Find the index of the root value in the inorder array
        inorder_index = inorder.index(root_val)

        # Build the right subtree first, then the left subtree
        # (This is because the postorder array is processed from the end)
        root.right = self.buildTree(inorder[inorder_index+1:], postorder)
        root.left = self.buildTree(inorder[:inorder_index], postorder)

        return root
    

# Q107
# Queue: Used in BFS ALOT (FIFO)
# node is an instance of the TreeNode class, which contains more information than just its value. 
# It includes attributes like val, left, and right. Thus in line XXX, we got to do node.val not node even if it is popped
class TreeNode:
    def __init__(self, val, left, right):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def levelOrderBottom(self, root):
        res = []
        if not root: return res

        queue = [root]

        while queue:
            val_at_level = []

            for _ in range(len(queue)):
                node = queue.pop(0)
                val_at_level.append(node.val)

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            res.append(val_at_level)
                
        res.reverse()
        return res
    
# Q108
# Height balanced subtree: Meaning depth of two subtrees of a node never differs > 1
# This means that subtree depth cannot be 0 and 2
class TreeNode:
    def __init__(self, val, left, right):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def sortedArrayToBST(self, nums):
        def convert(left, right):
            # left, right = 0, len(nums)-1
            if left>right: return None

            middle = (left+right)//2

            node = TreeNode(nums[middle])

            node.left = convert(left, middle-1)
            node.right = convert(middle+1, right)

            return node
        
        return convert(0, len(nums)-1)
    

# Q109
class Solution:
    def sortedListToBST(self, head: ListNode) -> TreeNode:
        if not head:
            return None
        if not head.next:
            return TreeNode(head.val)
        middle = self.getMiddle(head)
        root = TreeNode(middle.val)
        root.right = self.sortedListToBST(middle.next)
        middle.next = None
        root.left = self.sortedListToBST(head)
        return root
    
    def getMiddle(self, head: ListNode) -> ListNode:
        fast = head
        slow = head
        prev = None
        while fast and fast.next:
            fast = fast.next.next
            prev = slow
            slow = slow.next
        if prev:
            prev.next = None
        return slow
    
# Q110
# height balanced binary tree: depth of the two subtrees of every node never differs by more than one
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        def check_height(node):
            if not node:
                return 0
            
            left_height = check_height(node.left)
            right_height = check_height(node.right)
            
            if left_height == -1 or right_height == -1:
                return -1
            if abs(left_height - right_height) > 1:
                return -1

            return 1 + max(left_height, right_height)
        
        return check_height(root) != -1
    

# Q111
class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if not root: return 0

        left_depth = self.minDepth(root.left)
        right_depth = self.minDepth(root.right)

        if not left_depth and not right_depth: return 1

        if not left_depth:
            return right_depth + 1

        if not right_depth:
            return left_depth + 1

        return min(left_depth, right_depth) + 1
        

# Q112
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if not root:
            return False

        if not root.left and not root.right and root.val == targetSum:
            return True

        return self.hasPathSum(root.left, targetSum-root.val) or self.hasPathSum(root.right, targetSum-root.val)


# Q113
class Solution:        
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        if root is None:
            return []
        q, paths = deque([(root, targetSum, [])]), []
        
        while q:
            cur, target, path = q.pop()  
            if not (cur.left or cur.right) and cur.val == target:
                paths.append(path + [cur.val])
            else:
                if cur.left:
                    q.appendleft((cur.left, target - cur.val, path + [cur.val]))
                if cur.right:
                    q.appendleft((cur.right, target - cur.val, path + [cur.val]))
                                 
        return paths


# Q114
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        curr=root
        while curr:
            if curr.left!=None:
                prev=curr.left
                while prev.right:
                    prev=prev.right
                prev.right=curr.right
                curr.right=curr.left
                curr.left=None
            curr=curr.right

    
# Q118
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        if numRows == 1: return [[1]]
        if numRows == 2: return [[1],[1,1]]
        
        total = [[1],[1,1]]

        for i in range(2, numRows):
            prev_row = total[-1]
            new_row  = [1]

            for j in range(1, len(prev_row)):
                new_row.append(prev_row[j-1] + prev_row[j])
            
            new_row.append(1)
            total.append(new_row)

        return total
    

# Q119
class Solution:
    def getRow(self, rowIndex: int) -> List[int]:
        
        row = [1]

        for i in range(1, rowIndex+1):
            new_row = [1]

            for j in range(1,i):
                new_element = row[j-1] + row[j]
                new_row.append(new_element)
            
            new_row.append(1)

            row = new_row
        return row
        
    
# Q120
# Dynamic Programming with O(n) Space Complexity
# The method for this prob is to calculating from bottom to top to avoid repetition
# If I'm at triangle[i][j], the min path sum is tri[i][j] + min(tri[i+1][j], tri[i+1][j+1])
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        # if not triangle(1): return triangle(0)

        dp = triangle[-1]

        for i in range(len(triangle)-2, -1, -1):
            for j in range(len(triangle[i])):
                dp[j] = triangle[i][j] + min(dp[j], dp[j+1])

        return dp[0]


# Q121
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        min_price = prices[0]
        max_profit = 0
        
        for price in prices[1:]:
            max_profit = max(max_profit, price - min_price)
            min_price = min(min_price, price)
            
        return max_profit


# Q122
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        max = 0
        start = prices[0]
        len1 = len(prices)
        for i in range(0 , len1):
            if start < prices[i]: 
                max += prices[i] - start
            start = prices[i]
        return max
    

# Q125
class Solution:
    def isPalindrome(self, s: str) -> bool:
        i, j = 0, len(s) - 1
        while i < j:
            while i < j and not s[i].isalnum(): i += 1
            while i < j and not s[j].isalnum(): j -= 1

            if s[i].lower() != s[j].lower(): return False
            i += 1
            j -= 1

        return True


# Q 128
# must write algorithm in O(n): implies the solution should process the array in linear time 
# sorting is usually O(nlogn), and also there cannot be nested manners
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        ordered = set(nums)
        longest_streak = 0

        for num in ordered:
            if num-1 not in ordered:
                current = num
                streak = 1

                while current + 1 in ordered:
                    current += 1
                    streak += 1
                
                longest_streak = max(longest_streak, streak)


        return longest_streak
    

# Q129
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        def dfs(node, current_sum):
            if not node:
                return 0
            
            current_sum = current_sum * 10 + node.val

            if not node.left and not node.right:# meaning leaf node
                return current_sum

            left_sum = dfs(node.left, current_sum)
            right_sum = dfs(node.right, current_sum)

            return left_sum + right_sum

        return dfs(root, 0)


# Q130

        
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        if not board or not board[0]:
            return
        
        rows, cols = len(board), len(board[0])
        
        def dfs(r, c):
            # If out of bounds or current cell is not 'O', return
            if r < 0 or c < 0 or r >= rows or c >= cols or board[r][c] != 'O':
                return
            
            # Mark this cell as 'T' to indicate it's connected to the boundary
            board[r][c] = 'T'
            
            # Explore the neighboring cells (up, down, left, right)
            dfs(r+1, c)
            dfs(r-1, c)
            dfs(r, c+1)
            dfs(r, c-1)
        
        # Step 1: Start DFS from the boundary 'O' cells
        for r in range(rows):
            for c in range(cols):
                # Start from the first and last column for each row (boundary cells)
                if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                    if board[r][c] == 'O':
                        dfs(r, c)
        
        # Step 2: Flip all remaining 'O's to 'X', and turn 'T' back to 'O'
        for r in range(rows):
            for c in range(cols):
                if board[r][c] == 'O':
                    board[r][c] = 'X'  # These are surrounded regions
                elif board[r][c] == 'T':
                    board[r][c] = 'O'  # These are boundary-connected regions


# Q131
# Palindrome question: use backtracking!
class Solution:
    def partition(self, s: str) -> List[List[str]]:


    #### OR ANOTHER SOLUTION WOULD BE BELOW:
        result = []

        def backtrack(start, path):
            # If we have reached the end of the string, append the current partition to the result
            if start == len(s):
                result.append(path[:])
                return
            
            # Try to partition the string at every possible point
            for end in range(start + 1, len(s) + 1):
                current_substring = s[start:end]
                # Check if the current substring is a palindrome
                if current_substring == current_substring[::-1]:
                    # If it is a palindrome, add it to the current path
                    path.append(current_substring)
                    # Recur for the remaining substring
                    backtrack(end, path)
                    # Backtrack and remove the last substring
                    path.pop()

        # Initial call to the backtracking function
        backtrack(0, [])
        return result


# Q 133
"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""

from typing import Optional
class Solution:
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        if not node:
            return None
        
        # Dictionary to keep track of cloned nodes
        visited = {}
        
        # Define a helper function for DFS
        def dfs(node):
            if node in visited:
                return visited[node]
            
            # Clone the node
            clone = Node(node.val)
            visited[node] = clone
            
            # Clone all the neighbors recursively
            for neighbor in node.neighbors:
                clone.neighbors.append(dfs(neighbor))
            
            return clone
        
        # Start the DFS from the given node
        return dfs(node)


# Q134
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        
        sum_cost = sum(cost)
        sum_gas = sum(gas)

        if sum_cost > sum_gas:
            return -1

        current_gas = 0
        starting_index = 0

        for i in range(len(gas)):
            current_gas += gas[i] - cost[i]
            if current_gas < 0:
                current_gas = 0
                starting_index = i + 1
        return starting_index
        

# Q136
# question asks for linear runtime complexity, only constant extra space
# --> O(n) time, O(1) space
 ## solve by using bitwise XOR due to complexity requirements
# a ^ a = 0, a ^ b = 1, x ^ 0 = x
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        result = 0

        for num in nums:
            result = result ^ num

        return result
    
# Q137
from collections import Counter
# above will significantly make things easier and improve time complexity
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        count = Counter(nums)  # Create a dictionary with counts of each number
        for num, freq in count.items():
            if freq == 1:  # Return the number that appears exactly once
                return num



# Q138
class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        if not head:
            return None
        
        curr = head
        while curr:
            new_node = Node(curr.val, curr.next)
            curr.next = new_node
            curr = new_node.next
            
        curr = head
        while curr:
            if curr.random:
                curr.next.random = curr.random.next
            curr = curr.next.next
        
        old_head = head
        new_head = head.next
        curr_old = old_head
        curr_new = new_head
        
        while curr_old:
            curr_old.next = curr_old.next.next
            curr_new.next = curr_new.next.next if curr_new.next else None
            curr_old = curr_old.next
            curr_new = curr_new.next
            
        return new_head
    

# Q139
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        
        def construct(current,wordDict, memo={}):
            if current in memo:
                return memo[current]

            if not current:
                return True

            for word in wordDict:
                if current.startswith(word):
                    new_current = current[len(word):]
                    if construct(new_current,wordDict,memo):
                        memo[current] = True
                        return True

            memo[current] = False
            return False

        return construct(s,wordDict)