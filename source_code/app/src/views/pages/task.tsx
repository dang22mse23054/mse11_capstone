import React, { Component } from 'react';
import initWrapper from 'compDir/Wrapper';
import TaskList from 'compDir/pages/Task/List';
const TaskListWrapper = initWrapper(TaskList);

interface IState {
}

interface IProps {
}

class TaskPage extends Component<IProps, IState> {
	// Set default properties's values
	public static defaultProps: IProps = {
	}

	// Set default state
	public state: IState = {
	}

	public render(): React.ReactNode {
		return <TaskListWrapper {...this.props} />;
	}
}

export default TaskPage;