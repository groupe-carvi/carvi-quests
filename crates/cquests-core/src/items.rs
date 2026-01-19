//! item type and implementations for CQuests.
//!
//! Current state: implementation

use rand_chacha::ChaCha20Rng;

id_newtype!(ItemId);


#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
/// Item base
pub struct Item {
    pub id: ItemId,
    pub name: String,
    pub item_type: ItemType,
    pub weight: f32
}

// TODO - Add type specific data
pub enum ItemType {
    Consumable,
    Equipment,
}

impl Item {
    // Could use an error check as an output
    pub fn new(id: ItemId, name: String, item_type: ItemType, weight: f32) -> Self {
        Self {
            id,
            name,
            item_type,
            weight
        }
    }
}