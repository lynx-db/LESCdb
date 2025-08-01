#!/usr/bin/env python3
from typing import Any, Generic, Optional, TypeVar, final
import pickle
import os

K = TypeVar('K')
V = TypeVar('V')

@final
class BTreeNode(Generic[K, V]):
    def __init__(self, leaf: bool = False, order: int = 5):
        self.leaf = leaf
        self.order = order
        self.keys: list[K] = []
        self.values: list[V] = []
        self.children: list[BTreeNode[K, V]] = []

    def is_full(self) -> bool:
        return len(self.keys) >= 2 * self.order - 1

@final
class BTree(Generic[K, V]):
    def __init__(self, order: int = 5):
        self.root: BTreeNode[K, V] = BTreeNode(leaf=True, order=order)
        self.order = order

    def search(self, key: K) -> Optional[V]:
        return self._search_node(self.root, key)

    def _search_node(self, node: BTreeNode[K, V], key: K) -> Optional[V]:
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1

        if i < len(node.keys) and key == node.keys[i]:
            return node.values[i]
        elif node.leaf:
            return None
        else:
            return self._search_node(node.children[i], key)

    def insert(self, key: K, value: V) -> None:
        root = self.root
        if root.is_full():
            # Create new root
            new_root = BTreeNode(leaf=False, order=self.order)
            new_root.children.append(self.root)
            self.root = new_root
            self._split_child(new_root, 0)
        self._insert_non_full(self.root, key, value)

    def _split_child(self, parent: BTreeNode[K, V], index: int) -> None:
        order = self.order
        child = parent.children[index]
        new_child = BTreeNode(leaf=child.leaf, order=self.order)

        # Move the upper half of keys and values to new_child
        mid = order - 1
        new_child.keys = child.keys[mid+1:]
        new_child.values = child.values[mid+1:]
        
        # If not leaf, move the children too
        if not child.leaf:
            new_child.children = child.children[mid+1:]
            child.children = child.children[:mid+1]

        # Insert the middle key and value into parent
        parent.keys.insert(index, child.keys[mid])
        parent.values.insert(index, child.values[mid])
        parent.children.insert(index + 1, new_child)

        # Truncate child's keys and values
        child.keys = child.keys[:mid]
        child.values = child.values[:mid]

    def _insert_non_full(self, node: BTreeNode[K, V], key: K, value: V) -> None:
        i = len(node.keys) - 1

        if node.leaf:
            # Insert key in the correct position
            while i >= 0 and key < node.keys[i]:
                i -= 1
            node.keys.insert(i + 1, key)
            node.values.insert(i + 1, value)
        else:
            # Find the child which will have the new key
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1

            # If the child is full, split it
            if node.children[i].is_full():
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1

            self._insert_non_full(node.children[i], key, value)

    def delete(self, key: K) -> bool:
        return self._delete(self.root, key)

    def _delete(self, node: BTreeNode[K, V], key: K) -> bool:
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1

        # Key found in this node
        if i < len(node.keys) and key == node.keys[i]:
            if node.leaf:
                # Case 1: Key is in leaf node - just remove it
                node.keys.pop(i)
                node.values.pop(i)
                return True
            else:
                # Case 2: Key is in internal node
                return self._delete_internal_node(node, i)
        elif node.leaf:
            # Key not found in tree
            return False
        else:
            # Key might be in child node
            return self._delete_from_subtree(node, i, key)

    def _delete_internal_node(self, node: BTreeNode[K, V], index: int) -> bool:
        key = node.keys[index]
        
        # Case 2a: Predecessor has at least order keys
        if len(node.children[index].keys) >= self.order:
            # Find predecessor
            pred_node = node.children[index]
            while not pred_node.leaf:
                pred_node = pred_node.children[-1]
            
            # Replace key with predecessor
            pred_key = pred_node.keys[-1]
            pred_value = pred_node.values[-1]
            node.keys[index] = pred_key
            node.values[index] = pred_value
            
            # Delete predecessor from leaf
            return self._delete(node.children[index], pred_key)
            
        # Case 2b: Successor has at least order keys
        elif len(node.children[index + 1].keys) >= self.order:
            # Find successor
            succ_node = node.children[index + 1]
            while not succ_node.leaf:
                succ_node = succ_node.children[0]
                
            # Replace key with successor
            succ_key = succ_node.keys[0]
            succ_value = succ_node.values[0]
            node.keys[index] = succ_key
            node.values[index] = succ_value
            
            # Delete successor from leaf
            return self._delete(node.children[index + 1], succ_key)
            
        # Case 2c: Both children have minimum keys, merge them
        else:
            self._merge_children(node, index)
            return self._delete(node.children[index], key)

    def _delete_from_subtree(self, node: BTreeNode[K, V], index: int, key: K) -> bool:
        child = node.children[index]
        
        # If child has minimum number of keys
        if len(child.keys) == self.order - 1:
            # Try to borrow from left sibling
            if index > 0 and len(node.children[index - 1].keys) >= self.order:
                self._borrow_from_prev(node, index)
            # Try to borrow from right sibling
            elif index < len(node.children) - 1 and len(node.children[index + 1].keys) >= self.order:
                self._borrow_from_next(node, index)
            # Merge with a sibling
            else:
                if index < len(node.children) - 1:
                    self._merge_children(node, index)
                else:
                    self._merge_children(node, index - 1)
                    index -= 1
        
        return self._delete(node.children[index], key)

    def _borrow_from_prev(self, node: BTreeNode[K, V], index: int) -> None:
        child = node.children[index]
        sibling = node.children[index - 1]
        
        # Move key from parent to child
        child.keys.insert(0, node.keys[index - 1])
        child.values.insert(0, node.values[index - 1])
        
        # Move last key from sibling to parent
        node.keys[index - 1] = sibling.keys[-1]
        node.values[index - 1] = sibling.values[-1]
        
        # If not leaf, move last child from sibling to child
        if not child.leaf:
            child.children.insert(0, sibling.children.pop())
            
        sibling.keys.pop()
        sibling.values.pop()

    def _borrow_from_next(self, node: BTreeNode[K, V], index: int) -> None:
        child = node.children[index]
        sibling = node.children[index + 1]
        
        # Move key from parent to child
        child.keys.append(node.keys[index])
        child.values.append(node.values[index])
        
        # Move first key from sibling to parent
        node.keys[index] = sibling.keys[0]
        node.values[index] = sibling.values[0]
        
        # If not leaf, move first child from sibling to child
        if not child.leaf:
            child.children.append(sibling.children.pop(0))
            
        sibling.keys.pop(0)
        sibling.values.pop(0)

    def _merge_children(self, node: BTreeNode[K, V], index: int) -> None:
        child = node.children[index]
        sibling = node.children[index + 1]
        
        # Add key from parent to child
        child.keys.append(node.keys[index])
        child.values.append(node.values[index])
        
        # Move all keys from sibling to child
        child.keys.extend(sibling.keys)
        child.values.extend(sibling.values)
        
        # If not leaf, move all children from sibling to child
        if not child.leaf:
            child.children.extend(sibling.children)
            
        # Remove key and child pointer from parent
        node.keys.pop(index)
        node.values.pop(index)
        node.children.pop(index + 1)
        
        # If parent is now empty, make child the new root
        if not node.keys and node == self.root:
            self.root = child

@final
class BTreeDatabase(Generic[K, V]):
    def __init__(self, order: int = 5, path: str = "btree_db"):
        self.btree: BTree[K, V] = BTree(order=order)
        self.path = path
        self._load()
        
    def insert(self, key: K, value: V) -> None:
        """Insert a key-value pair into the database."""
        self.btree.insert(key, value)
        
    def search(self, key: K) -> Optional[V]:
        """Search for a value by key."""
        return self.btree.search(key)
        
    def delete(self, key: K) -> bool:
        """Delete a key-value pair by key."""
        return self.btree.delete(key)
    
    def keys(self) -> List[K]:
        """Return all keys in the database."""
        keys = []
        self._collect_keys(self.btree.root, keys)
        return keys
    
    def _collect_keys(self, node: BTreeNode[K, V], keys: List[K]) -> None:
        """Recursively collect all keys from a node and its children."""
        if not node:
            return
        
        # If not leaf, collect keys from first child
        if not node.leaf and node.children:
            self._collect_keys(node.children[0], keys)
            
        # Collect keys from this node and remaining children
        for i, key in enumerate(node.keys):
            keys.append(key)
            if not node.leaf and i + 1 < len(node.children):
                self._collect_keys(node.children[i + 1], keys)
    
    def save(self) -> None:
        """Save the database to disk."""
        db_file = f"{self.path}.db"
        with open(db_file, 'wb') as f:
            pickle.dump(self.btree, f)
            
    def _load(self) -> None:
        """Load the database from disk if it exists."""
        db_file = f"{self.path}.db"
        if os.path.exists(db_file):
            try:
                with open(db_file, 'rb') as f:
                    self.btree = pickle.load(f)
            except (pickle.PickleError, EOFError):
                print(f"Error loading database from {db_file}, creating new database")
                self.btree = BTree(order=self.btree.order)
  
    def clear(self) -> None:
        """Remove all entries from the database."""
        self.btree = BTree(order=self.btree.order)

# Example usage
if __name__ == "__main__":
    # Create a database
    db = BTreeDatabase[str, dict](path="example_db")
    
    # Insert some data
    db.insert("user1", {"name": "Alice", "age": 30})
    db.insert("user2", {"name": "Bob", "age": 25})
    db.insert("user3", {"name": "Charlie", "age": 35})
    
    # Search for data
    result = db.search("user2")
    if result:
        print(f"Found: {result}")
    
    # Save the database
    db.save()
    
    # Create a new instance that will load from the saved file
    new_db = BTreeDatabase[str, dict](path="example_db")
    
    # Search in the loaded database
    result = new_db.search("user1")
    if result:
        print(f"Loaded and found: {result}")
    
    # Delete an entry
    db.delete("user3")
    print(f"After deletion, user3 exists: {db.search('user3') is not None}")
    
    # Save changes
    db.save() 
