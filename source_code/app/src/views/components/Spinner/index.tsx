import React, { FunctionComponent } from 'react';
import { CircularProgress } from '@material-ui/core';

interface IProps {
	title?: string
	height?: number
	size?: number
}

const Spinner: FunctionComponent<IProps> = (props) => {
	const title = props.title ? props.title : 'Loading';
	const height = props.height ? props.height : 100;
	const size = props.size ? props.size : 20;

	return (
		<div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', height: height }} >
			<CircularProgress size={size} />
			<div style={{ textAlign: 'center', marginTop: 5 }}>{title}...</div>
		</div>
	);
};

export default Spinner;
