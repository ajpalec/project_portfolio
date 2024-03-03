import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.InOrder;
import org.mockito.Mockito;

class MasterControlTest {

	MasterControl masterControl;
	ScannerWrapper scannerWrapper;
	SystemWrapper systemWrapper;

	@BeforeEach
	void setUp() {
		masterControl = new MasterControl();
		scannerWrapper = Mockito.mock(ScannerWrapper.class);
		systemWrapper = Mockito.mock(SystemWrapper.class);
	}

	@Test
	void full_kwic() {
		Mockito.when(scannerWrapper.nextLine()).thenReturn("Descent of Man", "The Ascent of Man",
				"The Old Man and The Sea", "-1");
		masterControl.start(scannerWrapper, systemWrapper);
		InOrder inOrder = Mockito.inOrder(systemWrapper);
		inOrder.verify(systemWrapper).println("Please enter lines to add, then enter -1 to finish:");
		inOrder.verify(systemWrapper).println("and The Sea The Old Man");
		inOrder.verify(systemWrapper).println("Ascent of Man The");
		inOrder.verify(systemWrapper).println("Descent of Man");
		inOrder.verify(systemWrapper).println("Man and The Sea The Old");
		inOrder.verify(systemWrapper).println("Man Descent of");
		inOrder.verify(systemWrapper).println("Man The Ascent of");
		inOrder.verify(systemWrapper).println("of Man Descent");
		inOrder.verify(systemWrapper).println("of Man The Ascent");
		inOrder.verify(systemWrapper).println("Old Man and The Sea The");
		inOrder.verify(systemWrapper).println("Sea The Old Man and The");
		inOrder.verify(systemWrapper).println("The Ascent of Man");
		inOrder.verify(systemWrapper).println("The Old Man and The Sea");
		inOrder.verify(systemWrapper).println("The Sea The Old Man and");
		inOrder.verifyNoMoreInteractions();

	}

}
