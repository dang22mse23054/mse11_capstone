import React, { Component } from 'react';

interface IState {
}

interface IProps {
	message?: string;
	redirectUrl?: string;
}

export class LoadingPage extends Component<IProps, IState> {
	// Set default properties's values
	public static defaultProps: Partial<IProps> = {
		message: 'ローディング'
	}

	// Set default state
	public state: IState = {
	}

	componentDidMount() {
		// do something meaningful, Promises, if/else, whatever, and then
		if (this.props.redirectUrl) {
			window.location.assign(this.props.redirectUrl);
		}
	}

	public render(): React.ReactNode {
		return <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }} >
			<div className="custom-spinner"></div>
			<div style={{ marginInline: 10, textAlign: 'center' }}>{this.props.message}</div>
		</div>;
	}
}