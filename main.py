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
        


# 4
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