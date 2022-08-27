## 2214

```python
class Solution:
    def minimumHealth(self, damage: List[int], armor: int) -> int:
        max_damage = max(damage)
        min_health = sum(damage) - min(max_damage, armor) + 1
        
        return min_health
```



## 2281