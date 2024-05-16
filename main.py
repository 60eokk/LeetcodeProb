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