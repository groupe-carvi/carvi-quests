//! Inventory system core for for CQuests.
//! 
//! Current state: implementation

pub struct Inventory<T> {
    // Could be a player, NPC or container-type item
    pub ownerId: T,
    pub item_list: Hashmap<ItemId, String>,
    pub capacity: f32,
}

impl Inventory {
    pub fn new<T>(ownerId: T, max_capacity: f32) -> Self {
        let item_list: Hashmap<ItemId, String> = Hashmap::new();

        Self {
            ownerId,
            item_list,
            capacity: max_capacity
        }
    }

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
    }
}

mod test {
    use super::*;

    #[test]
    fn item_added() {
        let test_item = Item::new(
            123,
            "The sword of a thousand truth",
            Equipment,
            10
        );

        let test_inventrory = Inventory::new(
            1,
            11.0
        );

        test_inventrory.add_item(test_item);
    }
}