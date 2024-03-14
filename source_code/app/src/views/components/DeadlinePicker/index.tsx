import DeadlinePicker, { IDispatchToProps, IStateToProps } from './DeadlinePicker';
import { connect } from 'react-redux';
import { Actions } from 'servDir/redux/actions';

function mapStateToProps(store): IStateToProps {
	const scheduleReducer = store.scheduleReducer;
	const scheduleInfo: ISchedule = scheduleReducer.setting;

	return {
		type: scheduleInfo.type,
		deadlines: scheduleInfo.deadlines
	};
}

function mapDispatchToProps(dispatch, ownProps): IDispatchToProps {
	return {
		initData: async (_component: Page) => {
			return true;
		},

		onAdd: () => {
			dispatch(Actions.ScheduleAction.addDeadline());
		},

		onChange: (order: number, time: Moment) => {
			dispatch(Actions.ScheduleAction.changeDeadline(order, time));
		},

		onDelete: (order: number) => {
			dispatch(Actions.ScheduleAction.deleteDeadline(order));
		},
	};
}

export default connect(mapStateToProps, mapDispatchToProps)(DeadlinePicker);