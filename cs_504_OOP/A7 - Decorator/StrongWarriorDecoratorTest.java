import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class StrongWarriorDecoratorTest {

	@Test
	void correct_instance() {
		Warrior warrior = new StrongWarriorDecorator(new AggressiveWarrior.Builder(1).build());
		assertTrue(warrior instanceof StrongWarriorDecorator);
	}

	@Test
	void double_attack_aggressive() {
		Warrior warrior = new StrongWarriorDecorator(new AggressiveWarrior.Builder(1).attack(10).build());
		assertSame(20, warrior.getAttack());
	}

	@Test
	void defense_unchanged_aggressive() {
		Warrior warrior = new StrongWarriorDecorator(new AggressiveWarrior.Builder(1).defense(5).attack(10).build());
		assertSame(5, warrior.getDefense());
	}

	@Test
	void double_attack_calculate_attack_aggressive() {
		Warrior warrior = new StrongWarriorDecorator(new AggressiveWarrior.Builder(1).attack(10).build());
		assertSame(22, warrior.calculateAttack());
	}

	@Test
	void double_attack_calculate_boost_aggressive() {
		Warrior warrior = new StrongWarriorDecorator(new AggressiveWarrior.Builder(1).attack(10).build());
		assertEquals(10, warrior.calculateBoost());
	}

	@Test
	void double_attack_get_power_aggressive() {
		Warrior warrior = new StrongWarriorDecorator(new AggressiveWarrior.Builder(1).attack(10).build());
		assertEquals(35, warrior.calculatePower());
	}

	@Test
	void double_attack_defensive() {
		Warrior warrior = new StrongWarriorDecorator(new DefensiveWarrior.Builder(1).attack(10).build());
		assertSame(20, warrior.getAttack());
	}

	@Test
	void double_attack_calculate_attack_defensive() {
		Warrior warrior = new StrongWarriorDecorator(new DefensiveWarrior.Builder(1).attack(10).build());
		assertSame(21, warrior.calculateAttack());
	}

	@Test
	void double_attack_get_power_defensive() {
		Warrior warrior = new StrongWarriorDecorator(new DefensiveWarrior.Builder(1).attack(10).build());
		assertEquals(27.5, warrior.calculatePower());
	}
}
