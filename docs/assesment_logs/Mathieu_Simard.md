# Interview log

## Preparation/Brainstorming

I had a few ideas on what to add to the inventory system. My main problem was the how.
This kind of project is bigger than what i'm used to and I never programmed any games.
I also don't have much experience in Rust, so my code is probably not idiomatic.

I decided that the inventory system should have two parts :

1. Items, which is the general item system for the games.
2. Inventory, that control how items are related to a player, NPC, or a container object.

## Implementation

For the iplementation, I first clarified with Vincent what were the responsibilities of the crates where my code would go.

### Items

An item has an id, a name, a type and a weight.

The item type using an enum makes it possible to implement the desired behavior associated with said type.

For example; an item can be and equipment or a consumable. They can be unique or stackable, etc.

### Inventory

The inventory is simple. It has :

- an owner
- an item list
- and starting capacity

The `Inventory::new()` function takes and owner id and a starting capacity, and creates an empty item list.

The list is an hastable containing an item id as a key and it's name as a value. It minimize the inventory size (pun intended) and those are realy the only necessary elements of the item type that are necessary here.

I also added an add and remove item function.

The add function check if the current capacity is great enough to hold the item, and add it if it can.
I didn't think of an implementation for a full inventory handling at the time, but the program should be able to eventually tell the player his inventory is full.

The remove function check for the id in the inventory list and remove it if it's there.

> The `.or_insert()` method would cause a bug if the item is stackable, it's not te best way to do this.
>
> A similar error is in the delete function. It will delete a full stack. Not great for a currency item.

```rust
    pub fn add_item(&mut self, item: Item) {
        if self.capacity - item.weight > 0 {
            self.item_list.or_insert(item.id, item.name);
            self.capacity -= item.weight;
        }
        // Need a way to tell if item can't be added
    }

    pub fn delete_item(&mut self, item: Item) {
        if self.item_list.contains_key(item.id) {
            self.item_list(item.id);
            self.capacity += item.weight;
        }
        // Need a way to tell if item is not in inventory
    }-
```

## Conclusion

Sadly, I did not have the time to implement much of the things I thought about.
I only worked in the core crate of the project, my progess was not as fast as I would have liked to.

One thing I think I should have done is concentrate my efforts on the inventory system and make it flexible enough for a subsequent item implementation.
My efforts would not have been split between two subsystems.
